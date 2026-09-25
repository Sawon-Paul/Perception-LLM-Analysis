// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Params
/// @notice Every tunable number in the system, changeable only by consortium vote.
///
/// Plan section 7.2. k, N, T, rewards, stake and the rest live here rather than
/// as constants in each contract, so the thesis can run a sensitivity test by
/// sending transactions instead of redeploying.
///
/// Changing a value needs approvals from 3 of the 4 consortium members. That is
/// the on-chain governance the thesis argues for: no single organisation can
/// quietly make fraud cheaper by lowering the stake, or make confirmation
/// easier by lowering k.
contract Params {
    uint8 public constant APPROVALS_REQUIRED = 3;

    mapping(bytes32 => uint256) private _values;
    mapping(bytes32 => bool) private _isSet;

    /// consortium members who may approve changes
    mapping(address => bool) public isConsortium;
    uint8 public consortiumCount;

    /// proposal id => member => approved
    mapping(bytes32 => mapping(address => bool)) public approved;
    /// proposal id => number of approvals so far
    mapping(bytes32 => uint8) public approvalCount;

    event ParamChanged(string name, uint256 oldValue, uint256 newValue);
    event ChangeProposed(bytes32 indexed proposalId, string name, uint256 value, address by);
    event ChangeApproved(bytes32 indexed proposalId, address by, uint8 count);

    error NotConsortium();
    error AlreadyApproved();
    error UnknownParam(string name);

    modifier onlyConsortium() {
        if (!isConsortium[msg.sender]) revert NotConsortium();
        _;
    }

    /// @param members the four validator organisations
    constructor(address[] memory members) {
        for (uint256 i = 0; i < members.length; i++) {
            if (!isConsortium[members[i]]) {
                isConsortium[members[i]] = true;
                consortiumCount++;
            }
        }

        // Starting values from plan section 15. Each is a starting point to be
        // justified from data, not a final answer.
        _set("k_confirm", 3);
        _set("N_rewarded", 5);
        _set("T_reward_window", 24 hours);
        _set("T_pending", 72 hours);
        _set("m_checkback", 3);
        _set("stake", 5);
        _set("registration_bond", 25);
        _set("slash_percent", 100);
        _set("rep_threshold_unlearn", 200);
        _set("match_radius_m", 15);
        _set("checkback_radius_m", 20);
        _set("max_reports_per_hour", 20);
        // reward per position, 1-indexed
        _set("reward_1", 10);
        _set("reward_2", 6);
        _set("reward_3", 4);
        _set("reward_4", 2);
        _set("reward_5", 1);
    }

    function _set(string memory name, uint256 value) private {
        bytes32 key = keccak256(bytes(name));
        uint256 old = _values[key];
        _values[key] = value;
        _isSet[key] = true;
        emit ParamChanged(name, old, value);
    }

    /// @notice Read a parameter. Reverts on an unknown name rather than
    /// returning 0, because a silent zero stake or zero k would disable a
    /// safety rule without anyone noticing.
    function get(string calldata name) external view returns (uint256) {
        bytes32 key = keccak256(bytes(name));
        if (!_isSet[key]) revert UnknownParam(name);
        return _values[key];
    }

    function exists(string calldata name) external view returns (bool) {
        return _isSet[keccak256(bytes(name))];
    }

    /// @notice Reward for a 1-indexed confirmation position, 0 past the table.
    function rewardFor(uint256 position) external view returns (uint256) {
        if (position == 0 || position > 5) return 0;
        bytes32 key = keccak256(
            abi.encodePacked("reward_", bytes1(uint8(48 + position)))
        );
        return _values[key];
    }

    function proposalId(string calldata name, uint256 value)
        public
        pure
        returns (bytes32)
    {
        return keccak256(abi.encode(name, value));
    }

    /// @notice Approve a change. The third approval applies it.
    /// A member cannot approve the same proposal twice, so one organisation
    /// cannot reach the threshold alone.
    function approveParam(string calldata name, uint256 value)
        external
        onlyConsortium
    {
        bytes32 pid = proposalId(name, value);
        if (approved[pid][msg.sender]) revert AlreadyApproved();

        approved[pid][msg.sender] = true;
        uint8 count = approvalCount[pid] + 1;
        approvalCount[pid] = count;

        if (count == 1) emit ChangeProposed(pid, name, value, msg.sender);
        emit ChangeApproved(pid, msg.sender, count);

        if (count >= APPROVALS_REQUIRED) {
            _set(name, value);
            delete approvalCount[pid];
        }
    }

    /// @notice Withdraw an approval before the threshold is reached.
    function revokeApproval(string calldata name, uint256 value)
        external
        onlyConsortium
    {
        bytes32 pid = proposalId(name, value);
        if (!approved[pid][msg.sender]) return;
        approved[pid][msg.sender] = false;
        approvalCount[pid] -= 1;
    }
}
