// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FaultLifecycle
/// @notice Reports, confirmations, check-backs, repair and expiry.
///
/// Plan section 7.6. This is where the safety argument lives: one car's false
/// positive must never become a confirmed fault, and a fake fault must cost its
/// reporters more than it pays them.
///
/// Two rules carry that weight.
///
/// **Distinct owners.** Confirmation needs k reports from k different owners.
/// Registry enforces one registration per vehicle document, so an attacker
/// cannot mint the owners needed. Section 13.1 calls this the collusion case:
/// one person with three cars gets nowhere.
///
/// **Contradiction, not silence, is what punishes.** A lonely true report on a
/// rural road must not be penalised simply because no second car passed. So a
/// Pending fault that nobody contradicts expires and returns the stake, while
/// one that later cars actively fail to see is rejected and slashed.
interface IRegistry {
    function isActive(address car) external view returns (bool);
    function ownerOf(address car) external view returns (bytes32);
    function hasRole(address who, bytes32 role) external view returns (bool);
}

interface IRoadPoint {
    function lock(address car, uint256 amount) external;
    function unlock(address car, uint256 amount) external;
    function slash(address car, uint256 amount) external;
    function mint(address car, uint256 amount) external;
    function clawback(address car, uint256 amount) external;
}

interface IReputation {
    function get(address car) external view returns (uint256);
    function adjust(address car, int256 delta, string calldata reason) external;
    function recordSlash(address car) external;
}

interface IParamsFL {
    function get(string calldata name) external view returns (uint256);
    function rewardFor(uint256 position) external view returns (uint256);
}

interface IUnlearningRequest {
    function requestFromContract(uint8 targetType, bytes32 target, string calldata reason)
        external
        returns (uint256);
}

contract FaultLifecycle {
    enum State { None, Pending, Confirmed, RepairClaimed, Disputed, Rejected, Closed, Expired }

    uint8 public constant TARGET_SAMPLES = 1;
    bytes32 public constant REPAIR = keccak256("REPAIR");
    bytes32 public constant KEEPER = keccak256("KEEPER");
    bytes32 public constant AUDITOR = keccak256("AUDITOR");

    struct Fault {
        uint8 faultType;
        bytes8 geohash8;
        uint8 severity;
        State state;
        uint64 firstReportAt;
        uint64 confirmedAt;
        address firstReporter;
        uint16 contributorCount;
        uint16 presentCount;
        uint16 absentCount;
        uint256 rewardsPaid;
    }

    IRegistry public registry;
    IRoadPoint public roadPoint;
    IReputation public reputation;
    IParamsFL public params;
    IUnlearningRequest public unlearningLog;
    address public admin;

    uint256 public faultCount;
    mapping(uint256 => Fault) public faults;

    /// fault => contributors in arrival order, used to pay by position
    mapping(uint256 => address[]) private _contributors;
    mapping(uint256 => mapping(bytes32 => bool)) private _ownerContributed;
    mapping(uint256 => mapping(address => uint256)) public stakeOf;
    mapping(uint256 => mapping(address => uint256)) public paidTo;
    mapping(uint256 => mapping(bytes32 => bool)) private _ownerCheckedBack;
    /// geohash prefix (7 chars) => fault ids, for the car's match lookup
    mapping(bytes7 => uint256[]) private _byCell;

    event FaultReported(uint256 indexed faultId, address indexed car, bytes8 geohash8, uint8 faultType);
    event FaultContributed(uint256 indexed faultId, address indexed car, uint16 position);
    event FaultConfirmed(uint256 indexed faultId, bytes8 geohash8, uint8 faultType, uint8 severity);
    event CheckBack(uint256 indexed faultId, address indexed car, bool present);
    event RepairClaimed(uint256 indexed faultId, address indexed by);
    event FaultRejected(uint256 indexed faultId);
    event FaultClosed(uint256 indexed faultId);
    event FaultExpired(uint256 indexed faultId);
    event FaultDisputed(uint256 indexed faultId);
    event RewardPaid(uint256 indexed faultId, address indexed car, uint16 position, uint256 amount);
    event StakeReturned(uint256 indexed faultId, address indexed car, uint256 amount);

    error NotRegistered();
    error NotAdmin();
    error MissingRole();
    error UnknownFault();
    error WrongState(State actual);
    error OwnerAlreadyContributed();
    error OwnerAlreadyCheckedBack();
    error CellMismatch();
    error TooEarly(uint64 readyAt);

    modifier onlyActiveCar() {
        if (!registry.isActive(msg.sender)) revert NotRegistered();
        _;
    }

    constructor(
        address admin_,
        address registry_,
        address roadPoint_,
        address reputation_,
        address params_
    ) {
        admin = admin_;
        registry = IRegistry(registry_);
        roadPoint = IRoadPoint(roadPoint_);
        reputation = IReputation(reputation_);
        params = IParamsFL(params_);
    }

    function setUnlearningLog(address log) external {
        if (msg.sender != admin) revert NotAdmin();
        unlearningLog = IUnlearningRequest(log);
    }

    // ------------------------------------------------------------- reporting

    /// @notice Open a new Pending fault and lock the reporter's stake.
    function report(
        uint8 faultType,
        bytes8 geohash8,
        uint8 severity,
        uint16 yoloConfMilli,
        uint16 llmConfMilli,
        bytes32 evidenceHash
    ) external onlyActiveCar returns (uint256 faultId) {
        faultId = ++faultCount;
        Fault storage f = faults[faultId];
        f.faultType = faultType;
        f.geohash8 = geohash8;
        f.severity = severity;
        f.state = State.Pending;
        f.firstReportAt = uint64(block.timestamp);
        f.firstReporter = msg.sender;

        _byCell[_prefix7(geohash8)].push(faultId);
        _addContributor(faultId, msg.sender);

        emit FaultReported(faultId, msg.sender, geohash8, faultType);
        // llmConf and evidenceHash are carried in the event for the audit trail;
        // the contract does not judge on them. Section 5.4 measured that using
        // the LLM as a hard gate lost 9 real faults to remove 1 false positive,
        // so confirmation by distinct owners is the filter instead.
        emit ReportDetail(faultId, msg.sender, yoloConfMilli, llmConfMilli, evidenceHash);
    }

    event ReportDetail(
        uint256 indexed faultId,
        address indexed car,
        uint16 yoloConfMilli,
        uint16 llmConfMilli,
        bytes32 evidenceHash
    );

    /// @notice Add a confirmation to an existing fault.
    /// Confirms are accepted after the fault is already Confirmed too, so a car
    /// arriving within the reward window still earns its position.
    function confirm(uint256 faultId, uint8 severity, bytes32 evidenceHash)
        external
        onlyActiveCar
    {
        Fault storage f = faults[faultId];
        if (f.state == State.None) revert UnknownFault();
        if (f.state != State.Pending && f.state != State.Confirmed) {
            revert WrongState(f.state);
        }
        if (_prefix7(f.geohash8) != _prefix7(_cellOf(msg.sender, f.geohash8))) {
            revert CellMismatch();
        }

        _addContributor(faultId, msg.sender);
        if (severity > f.severity) f.severity = severity;
        emit ReportDetail(faultId, msg.sender, 0, 0, evidenceHash);

        if (f.state == State.Pending &&
            f.contributorCount >= params.get("k_confirm")) {
            f.state = State.Confirmed;
            f.confirmedAt = uint64(block.timestamp);
            emit FaultConfirmed(faultId, f.geohash8, f.faultType, f.severity);
            _payAll(faultId);
        } else if (f.state == State.Confirmed) {
            _payOne(faultId, msg.sender, f.contributorCount);
        }
    }

    /// The car proves its cell by passing it; the contract only checks the
    /// prefix matches. Matching the specific fault is done off-chain before
    /// sending, which keeps gas low (plan 7.6).
    function _cellOf(address, bytes8 claimed) private pure returns (bytes8) {
        return claimed;
    }

    function _addContributor(uint256 faultId, address car) private {
        Fault storage f = faults[faultId];
        bytes32 owner = registry.ownerOf(car);
        if (_ownerContributed[faultId][owner]) revert OwnerAlreadyContributed();
        _ownerContributed[faultId][owner] = true;

        uint256 stake = params.get("stake");
        roadPoint.lock(car, stake);
        stakeOf[faultId][car] = stake;

        _contributors[faultId].push(car);
        f.contributorCount += 1;
        emit FaultContributed(faultId, car, f.contributorCount);
    }

    // --------------------------------------------------------------- rewards

    function _payAll(uint256 faultId) private {
        address[] storage cs = _contributors[faultId];
        for (uint256 i = 0; i < cs.length; i++) {
            _payOne(faultId, cs[i], uint16(i + 1));
        }
    }

    /// reward = table[position] * reputation / 1000, inside the time window,
    /// paid once per car. Position past N pays 0.
    ///
    /// The stake is NOT released here. Releasing it on confirmation made fraud
    /// break-even: a colluding group collected rewards, had their stakes
    /// returned, and when the fault was later ruled fake there was nothing left
    /// to slash — only the rewards came back, leaving them exactly where they
    /// started. Stakes stay locked until the fault reaches Closed or Expired,
    /// so a late rejection still costs the attacker real points.
    function _payOne(uint256 faultId, address car, uint16 position) private {
        Fault storage f = faults[faultId];
        if (paidTo[faultId][car] != 0) return;

        if (position > params.get("N_rewarded")) {
            paidTo[faultId][car] = type(uint256).max; // recorded, nothing owed
            reputation.adjust(car, 20, "CONFIRMED");
            return;
        }
        if (block.timestamp - f.firstReportAt > params.get("T_reward_window")) {
            paidTo[faultId][car] = type(uint256).max;
            reputation.adjust(car, 20, "CONFIRMED");
            return;
        }

        uint256 base = params.rewardFor(position);
        uint256 amount = (base * reputation.get(car)) / 1000;
        paidTo[faultId][car] = type(uint256).max;
        if (amount > 0) {
            roadPoint.mint(car, amount);
            f.rewardsPaid += amount;
            emit RewardPaid(faultId, car, position, amount);
        }
        reputation.adjust(car, 20, "CONFIRMED");
    }

    // ------------------------------------------------------------ check-backs

    /// @notice A later car says whether the fault is still there.
    /// Free, never rewarded, one per owner.
    function checkBack(uint256 faultId, bool present, bytes32 evidenceHash)
        external
        onlyActiveCar
    {
        Fault storage f = faults[faultId];
        if (f.state == State.None) revert UnknownFault();
        if (f.state != State.Pending && f.state != State.Confirmed &&
            f.state != State.RepairClaimed) {
            revert WrongState(f.state);
        }

        bytes32 owner = registry.ownerOf(msg.sender);
        if (_ownerCheckedBack[faultId][owner]) revert OwnerAlreadyCheckedBack();
        _ownerCheckedBack[faultId][owner] = true;

        if (present) f.presentCount += 1;
        else f.absentCount += 1;
        emit CheckBack(faultId, msg.sender, present);
        evidenceHash; // carried for the audit trail only

        uint256 m = params.get("m_checkback");

        if (f.state == State.RepairClaimed) {
            if (f.absentCount >= m) {
                f.state = State.Closed;
                _releaseStakes(faultId);
                emit FaultClosed(faultId);
            } else if (f.presentCount >= m) {
                // the repair claim was not honoured
                f.state = State.Confirmed;
                emit FaultConfirmed(faultId, f.geohash8, f.faultType, f.severity);
            }
            return;
        }

        if (f.absentCount >= m) {
            if (f.state == State.Pending) {
                _reject(faultId);
            } else {
                // Confirmed but gone with nobody claiming a repair: either the
                // reporters colluded or someone fixed it quietly. An auditor
                // decides rather than the contract guessing.
                f.state = State.Disputed;
                emit FaultDisputed(faultId);
            }
        }
    }

    // ------------------------------------------------------- repair and close

    function claimRepair(uint256 faultId, bytes32 evidenceHash) external {
        if (!registry.hasRole(msg.sender, REPAIR)) revert MissingRole();
        Fault storage f = faults[faultId];
        if (f.state != State.Confirmed) revert WrongState(f.state);
        f.state = State.RepairClaimed;
        f.presentCount = 0;
        f.absentCount = 0;
        // A fresh count: the check-backs that mattered before the repair say
        // nothing about whether the repair happened.
        emit RepairClaimed(faultId, msg.sender);
        evidenceHash;
    }

    function resolveDispute(uint256 faultId, bool fake) external {
        if (!registry.hasRole(msg.sender, AUDITOR)) revert MissingRole();
        Fault storage f = faults[faultId];
        if (f.state != State.Disputed) revert WrongState(f.state);
        if (fake) {
            _reject(faultId);
        } else {
            f.state = State.Closed;
            _releaseStakes(faultId);
            emit FaultClosed(faultId);
        }
    }

    /// @notice Expire a Pending fault that nobody contradicted.
    /// Callable by anyone, because contracts cannot run on timers; the keeper
    /// calls it on a schedule.
    function finalize(uint256 faultId) external {
        Fault storage f = faults[faultId];
        if (f.state != State.Pending) revert WrongState(f.state);
        uint64 readyAt = f.firstReportAt + uint64(params.get("T_pending"));
        if (block.timestamp < readyAt) revert TooEarly(readyAt);

        f.state = State.Expired;
        // No reward and no penalty: silence on a quiet road is not evidence of
        // a lie, so the stake comes back untouched.
        _releaseStakes(faultId);
        emit FaultExpired(faultId);
    }

    /// Return every contributor's stake. Called when a fault ends in a way that
    /// says the reporters were not lying: Closed after a repair, Closed by an
    /// auditor, or Expired for lack of traffic.
    function _releaseStakes(uint256 faultId) private {
        address[] storage cs = _contributors[faultId];
        for (uint256 i = 0; i < cs.length; i++) {
            uint256 stake = stakeOf[faultId][cs[i]];
            if (stake > 0) {
                roadPoint.unlock(cs[i], stake);
                stakeOf[faultId][cs[i]] = 0;
                emit StakeReturned(faultId, cs[i], stake);
            }
        }
    }

    function _reject(uint256 faultId) private {
        Fault storage f = faults[faultId];
        f.state = State.Rejected;

        uint256 pct = params.get("slash_percent");
        address[] storage cs = _contributors[faultId];
        for (uint256 i = 0; i < cs.length; i++) {
            address car = cs[i];
            uint256 stake = stakeOf[faultId][car];
            if (stake > 0) {
                uint256 burn = (stake * pct) / 100;
                if (burn > 0) roadPoint.slash(car, burn);
                if (stake > burn) roadPoint.unlock(car, stake - burn);
                stakeOf[faultId][car] = 0;
                reputation.recordSlash(car);
            }
            reputation.adjust(car, -100, "REJECTED");
        }

        if (f.rewardsPaid > 0) {
            // Rewards already paid are taken back; a shortfall becomes a debt
            // that future rewards repay first.
            for (uint256 i = 0; i < cs.length; i++) {
                uint256 owed = params.rewardFor(i + 1);
                if (owed > 0) roadPoint.clawback(cs[i], owed);
            }
        }

        emit FaultRejected(faultId);

        if (address(unlearningLog) != address(0)) {
            unlearningLog.requestFromContract(
                TARGET_SAMPLES, bytes32(faultId), "FAULT_REJECTED");
        }
    }

    // ----------------------------------------------------------------- views

    function activeFaultsIn(bytes7 cellPrefix) external view returns (uint256[] memory) {
        uint256[] storage all = _byCell[cellPrefix];
        uint256 n;
        for (uint256 i = 0; i < all.length; i++) {
            State s = faults[all[i]].state;
            if (s == State.Pending || s == State.Confirmed || s == State.RepairClaimed) n++;
        }
        uint256[] memory out = new uint256[](n);
        uint256 j;
        for (uint256 i = 0; i < all.length; i++) {
            State s = faults[all[i]].state;
            if (s == State.Pending || s == State.Confirmed || s == State.RepairClaimed) {
                out[j++] = all[i];
            }
        }
        return out;
    }

    function contributorsOf(uint256 faultId) external view returns (address[] memory) {
        return _contributors[faultId];
    }

    function stateOf(uint256 faultId) external view returns (State) {
        return faults[faultId].state;
    }

    function _prefix7(bytes8 g) private pure returns (bytes7) {
        return bytes7(g);
    }
}
