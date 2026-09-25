// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Reputation
/// @notice One score per car, 0 to 1000, starting at 500.
///
/// Plan section 7.5. This is the single number the thesis claims ties the
/// blockchain to the AI: it scales token rewards and it weights the car's
/// contribution during FL aggregation. A car that reports honestly earns more
/// and counts for more in the model; a car caught lying earns less, counts for
/// less, and eventually has its influence removed entirely.
///
/// Adjustments are asymmetric on purpose. A confirmed report gains 20; a
/// rejected one loses 100. Fraud has to cost more than honesty pays, or
/// reporting fake faults becomes a profitable strategy once a car has enough
/// reputation to absorb the loss.
interface IUnlearningLogRequest {
    function requestFromContract(uint8 targetType, bytes32 target, string calldata reason)
        external
        returns (uint256);
}

interface IParamsView {
    function get(string calldata name) external view returns (uint256);
}

contract Reputation {
    uint256 public constant START = 500;
    uint256 public constant MAX = 1000;

    int256 public constant DELTA_CONFIRMED = 20;
    int256 public constant DELTA_REJECTED = -100;
    int256 public constant DELTA_CONTRADICTED = -10;

    uint8 public constant TARGET_CLIENT = 0;

    address public admin;
    IParamsView public params;
    IUnlearningLogRequest public unlearningLog;

    mapping(address => bool) public isController;
    mapping(address => uint256) private _score;
    mapping(address => bool) private _seen;
    mapping(address => uint8) public slashCount;
    mapping(address => bool) public unlearnRequested;

    event ScoreChanged(address indexed car, uint256 oldScore, uint256 newScore, string reason);
    event SlashRecorded(address indexed car, uint8 count);
    event UnlearnTriggered(address indexed car, string reason);
    event ControllerSet(address indexed who, bool enabled);

    error NotAdmin();
    error NotController();

    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAdmin();
        _;
    }
    modifier onlyController() {
        if (!isController[msg.sender]) revert NotController();
        _;
    }

    constructor(address admin_, address params_) {
        admin = admin_;
        params = IParamsView(params_);
    }

    function setController(address who, bool enabled) external onlyAdmin {
        isController[who] = enabled;
        emit ControllerSet(who, enabled);
    }

    function setUnlearningLog(address log) external onlyAdmin {
        unlearningLog = IUnlearningLogRequest(log);
    }

    /// @notice Current score. Unseen cars read as START rather than 0, so a new
    /// car is not treated as maximally untrustworthy before it has done
    /// anything.
    function get(address car) public view returns (uint256) {
        return _seen[car] ? _score[car] : START;
    }

    /// @notice Apply a change, clamped to [0, MAX].
    /// Triggers unlearning when the car drops below the threshold, so the model
    /// stops carrying the influence of a car the chain no longer trusts.
    function adjust(address car, int256 delta, string calldata reason)
        external
        onlyController
    {
        uint256 old = get(car);
        int256 next = int256(old) + delta;
        if (next < 0) next = 0;
        if (next > int256(MAX)) next = int256(MAX);

        _score[car] = uint256(next);
        _seen[car] = true;
        emit ScoreChanged(car, old, uint256(next), reason);

        uint256 threshold = params.get("rep_threshold_unlearn");
        if (uint256(next) < threshold) {
            _triggerUnlearn(car, "REPUTATION_BELOW_THRESHOLD");
        }
    }

    /// @notice Record a slash. Two slashes trigger unlearning regardless of
    /// score, because a car can be caught twice while still sitting above the
    /// threshold.
    function recordSlash(address car) external onlyController {
        uint8 count = slashCount[car] + 1;
        slashCount[car] = count;
        emit SlashRecorded(car, count);
        if (count >= 2) {
            _triggerUnlearn(car, "TWO_SLASHES");
        }
    }

    function _triggerUnlearn(address car, string memory reason) private {
        // Requested once. Repeating it would make the unlearning service redo
        // the same rollback every time the car's score moved.
        if (unlearnRequested[car]) return;
        unlearnRequested[car] = true;
        emit UnlearnTriggered(car, reason);
        if (address(unlearningLog) != address(0)) {
            unlearningLog.requestFromContract(
                TARGET_CLIENT, bytes32(uint256(uint160(car))), reason);
        }
    }

    /// @notice Weight for FL aggregation: score / 1000, scaled by 1e18 so the
    /// off-chain server can use it without floating point on-chain.
    function flWeight(address car) external view returns (uint256) {
        return (get(car) * 1e18) / MAX;
    }
}
