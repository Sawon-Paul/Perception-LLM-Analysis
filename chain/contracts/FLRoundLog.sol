// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FLRoundLog
/// @notice The on-chain record of each federated learning round.
///
/// Plan section 7.7. No weights and no gradients go on the chain \u2014 only
/// hashes. The files live off-chain; what the chain provides is that every car
/// can check the model it downloaded is the one the round actually agreed on.
///
/// This is the defence against the model-swap attack: a compromised server can
/// serve any file it likes, but the hash is written here before cars fetch it,
/// and a car that finds a mismatch refuses to load and logs the rejection.
///
/// Contributions are recorded per car so aggregation is auditable after the
/// fact, and so unlearning can find which rounds a car influenced.
interface IRegistryFL {
    function hasRole(address who, bytes32 role) external view returns (bool);
    function isActive(address car) external view returns (bool);
    function isExcludedFromFL(address car) external view returns (bool);
}

interface IReputationFL {
    function flWeight(address car) external view returns (uint256);
}

contract FLRoundLog {
    bytes32 public constant FL_SERVER = keccak256("FL_SERVER");

    enum RoundState { None, Open, Aggregated }

    struct Round {
        RoundState state;
        uint64 openedAt;
        uint64 aggregatedAt;
        bytes32 baseModelHash;     // what clients started from
        bytes32 globalModelHash;   // what they must end with
        uint32 contributors;
        uint32 rejected;
        string note;
    }

    struct Contribution {
        bytes32 updateHash;
        uint32 samples;
        uint64 weight;             // reputation weight at submission time
        uint64 at;
    }

    address public admin;
    IRegistryFL public registry;
    IReputationFL public reputation;

    uint256 public currentRound;
    mapping(uint256 => Round) public rounds;
    mapping(uint256 => address[]) private _contributors;
    mapping(uint256 => mapping(address => Contribution)) public contributionOf;
    /// which rounds a car contributed to, for unlearning lookups
    mapping(address => uint256[]) private _roundsOf;

    event RoundOpened(uint256 indexed round, bytes32 baseModelHash, uint64 at);
    event UpdateSubmitted(uint256 indexed round, address indexed car,
                          bytes32 updateHash, uint32 samples, uint64 weight);
    event UpdateRejected(uint256 indexed round, address indexed car, string reason);
    event RoundAggregated(uint256 indexed round, bytes32 globalModelHash,
                          uint32 contributors, uint64 at);
    event ModelHashMismatch(uint256 indexed round, address indexed car,
                            bytes32 expected, bytes32 got);

    error NotServer();
    error NotAdmin();
    error WrongState();
    error NotEligible(string reason);
    error AlreadySubmitted();
    error ZeroHash();

    modifier onlyServer() {
        if (!registry.hasRole(msg.sender, FL_SERVER)) revert NotServer();
        _;
    }

    constructor(address admin_, address registry_, address reputation_) {
        admin = admin_;
        registry = IRegistryFL(registry_);
        reputation = IReputationFL(reputation_);
    }

    function setReputation(address reputation_) external {
        if (msg.sender != admin) revert NotAdmin();
        reputation = IReputationFL(reputation_);
    }

    /// @notice Start a round from a named base model.
    function openRound(bytes32 baseModelHash, string calldata note)
        external
        onlyServer
        returns (uint256 round)
    {
        if (baseModelHash == bytes32(0)) revert ZeroHash();
        if (currentRound != 0 && rounds[currentRound].state == RoundState.Open) {
            revert WrongState();
        }
        round = ++currentRound;
        Round storage r = rounds[round];
        r.state = RoundState.Open;
        r.openedAt = uint64(block.timestamp);
        r.baseModelHash = baseModelHash;
        r.note = note;
        emit RoundOpened(round, baseModelHash, r.openedAt);
    }

    /// @notice A car records the hash of the update it sent to the server.
    /// The weight is read from Reputation here rather than supplied, so the
    /// server cannot inflate a client's influence after the fact.
    function submitUpdate(uint256 round, bytes32 updateHash, uint32 samples)
        external
    {
        Round storage r = rounds[round];
        if (r.state != RoundState.Open) revert WrongState();
        if (updateHash == bytes32(0)) revert ZeroHash();
        if (!registry.isActive(msg.sender)) revert NotEligible("inactive");
        if (registry.isExcludedFromFL(msg.sender)) revert NotEligible("unlearned");
        if (contributionOf[round][msg.sender].at != 0) revert AlreadySubmitted();

        uint64 w = uint64(reputation.flWeight(msg.sender) / 1e12);  // 1e18 -> 1e6
        contributionOf[round][msg.sender] = Contribution({
            updateHash: updateHash, samples: samples, weight: w,
            at: uint64(block.timestamp)
        });
        _contributors[round].push(msg.sender);
        _roundsOf[msg.sender].push(round);
        r.contributors += 1;
        emit UpdateSubmitted(round, msg.sender, updateHash, samples, w);
    }

    /// @notice The server records an update it declined, with a reason.
    /// Kept on-chain so a server cannot silently drop a client it dislikes.
    function rejectUpdate(uint256 round, address car, string calldata reason)
        external
        onlyServer
    {
        if (rounds[round].state != RoundState.Open) revert WrongState();
        rounds[round].rejected += 1;
        emit UpdateRejected(round, car, reason);
    }

    /// @notice Close the round and publish the hash of the resulting model.
    function aggregate(uint256 round, bytes32 globalModelHash)
        external
        onlyServer
    {
        Round storage r = rounds[round];
        if (r.state != RoundState.Open) revert WrongState();
        if (globalModelHash == bytes32(0)) revert ZeroHash();
        r.state = RoundState.Aggregated;
        r.globalModelHash = globalModelHash;
        r.aggregatedAt = uint64(block.timestamp);
        emit RoundAggregated(round, globalModelHash, r.contributors, r.aggregatedAt);
    }

    /// @notice A car reports that the file it downloaded does not match the
    /// hash on the chain. Anyone can read these; a burst of them is the signal
    /// that a server is serving something other than what was agreed.
    function reportMismatch(uint256 round, bytes32 got) external {
        emit ModelHashMismatch(round, msg.sender, rounds[round].globalModelHash, got);
    }

    /// @notice What a car must verify before loading a global model.
    function expectedModelHash(uint256 round) external view returns (bytes32) {
        return rounds[round].globalModelHash;
    }

    function contributorsOf(uint256 round) external view returns (address[] memory) {
        return _contributors[round];
    }

    /// @notice Rounds a car influenced \u2014 the range unlearning must replay.
    function roundsOf(address car) external view returns (uint256[] memory) {
        return _roundsOf[car];
    }

    function firstRoundOf(address car) external view returns (uint256) {
        uint256[] storage rs = _roundsOf[car];
        return rs.length == 0 ? 0 : rs[0];
    }
}
