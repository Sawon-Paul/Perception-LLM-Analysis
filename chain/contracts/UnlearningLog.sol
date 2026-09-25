// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title UnlearningLog
/// @notice Requests to remove influence from the model, and certificates that
/// the removal happened.
///
/// Plan sections 7.8 and 12.2. Most unlearning work reports metrics its own
/// authors computed. A certificate here binds the method, the round range and
/// the before/after model hashes, so a third party can fetch both models, hash
/// them, re-run the metric and check the claim without trusting the service
/// that made it.
///
/// Be precise about what that proves. It proves the named files existed and
/// hash to the recorded values, that the method and round range were declared
/// before the model was swapped, and that every car loaded the same cleaned
/// model. It does NOT prove the removal procedure was executed honestly, nor
/// that no residual influence survives. The contribution is auditable
/// unlearning, not provable unlearning \u2014 and the difference belongs in the
/// write-up.
interface IRegistryUL {
    function hasRole(address who, bytes32 role) external view returns (bool);
}

contract UnlearningLog {
    bytes32 public constant UNLEARN_SERVICE = keccak256("UNLEARN_SERVICE");
    bytes32 public constant AUDITOR = keccak256("AUDITOR");

    /// what is being removed
    enum Target { Client, Samples, Region }
    enum Status { Requested, InProgress, Certified, Failed }
    /// A: retrain from scratch without the data (exact, slow)
    /// B: roll back to a round and redo without it (approximate, fast)
    enum Method { None, RetrainFromScratch, RollbackAndRedo }

    struct Request {
        Target target;
        bytes32 subject;        // car address, fault id, or region key
        string reason;
        address requestedBy;
        uint64 requestedAt;
        Status status;
        uint256 certificateId;
    }

    struct Certificate {
        uint256 requestId;
        Method method;
        uint256 fromRound;
        uint256 toRound;
        bytes32 modelBefore;
        bytes32 modelAfter;
        uint32 samplesRemoved;
        uint32 metricBefore;    // attack success rate, per mille
        uint32 metricAfter;
        uint64 durationSec;
        uint64 issuedAt;
        address issuedBy;
        bool auditPassed;
        address auditedBy;
    }

    address public admin;
    IRegistryUL public registry;
    /// contracts allowed to raise requests automatically
    mapping(address => bool) public isRequester;

    uint256 public requestCount;
    uint256 public certificateCount;
    mapping(uint256 => Request) public requests;
    mapping(uint256 => Certificate) public certificates;
    mapping(bytes32 => uint256[]) private _bySubject;

    event UnlearnRequested(uint256 indexed id, Target target, bytes32 indexed subject,
                           string reason, address by);
    event UnlearnStarted(uint256 indexed id, Method method, uint256 fromRound);
    event UnlearnCertified(uint256 indexed id, uint256 indexed certId,
                           bytes32 modelBefore, bytes32 modelAfter,
                           uint32 metricBefore, uint32 metricAfter);
    event UnlearnFailed(uint256 indexed id, string reason);
    event CertificateAudited(uint256 indexed certId, address auditor, bool passed,
                             string note);
    event RequesterSet(address indexed who, bool enabled);

    error NotAdmin();
    error NotService();
    error NotAuditor();
    error NotRequester();
    error UnknownRequest();
    error WrongStatus(Status actual);
    error ZeroHash();
    error EmptyRange();

    modifier onlyService() {
        if (!registry.hasRole(msg.sender, UNLEARN_SERVICE)) revert NotService();
        _;
    }

    constructor(address admin_, address registry_) {
        admin = admin_;
        registry = IRegistryUL(registry_);
    }

    function setRequester(address who, bool enabled) external {
        if (msg.sender != admin) revert NotAdmin();
        isRequester[who] = enabled;
        emit RequesterSet(who, enabled);
    }

    /// @notice Raised by a person or authority.
    function request(uint8 target, bytes32 subject, string calldata reason)
        external
        returns (uint256 id)
    {
        return _request(Target(target), subject, reason, msg.sender);
    }

    /// @notice Raised automatically by Reputation or FaultLifecycle when a car
    /// is slashed twice or a fault is rejected. The signature matches what
    /// those contracts call.
    function requestFromContract(uint8 targetType, bytes32 target,
                                 string calldata reason)
        external
        returns (uint256)
    {
        if (!isRequester[msg.sender]) revert NotRequester();
        return _request(Target(targetType), target, reason, msg.sender);
    }

    function _request(Target t, bytes32 subject, string memory reason, address by)
        private
        returns (uint256 id)
    {
        id = ++requestCount;
        requests[id] = Request({
            target: t, subject: subject, reason: reason, requestedBy: by,
            requestedAt: uint64(block.timestamp), status: Status.Requested,
            certificateId: 0
        });
        _bySubject[subject].push(id);
        emit UnlearnRequested(id, t, subject, reason, by);
    }

    function start(uint256 id, uint8 method, uint256 fromRound)
        external
        onlyService
    {
        Request storage r = requests[id];
        if (r.requestedAt == 0) revert UnknownRequest();
        if (r.status != Status.Requested) revert WrongStatus(r.status);
        r.status = Status.InProgress;
        emit UnlearnStarted(id, Method(method), fromRound);
    }

    /// @notice Publish the certificate. The hashes are what an auditor rechecks.
    function certify(
        uint256 id,
        uint8 method,
        uint256 fromRound,
        uint256 toRound,
        bytes32 modelBefore,
        bytes32 modelAfter,
        uint32 samplesRemoved,
        uint32 metricBefore,
        uint32 metricAfter,
        uint64 durationSec
    ) external onlyService returns (uint256 certId) {
        Request storage r = requests[id];
        if (r.requestedAt == 0) revert UnknownRequest();
        if (r.status != Status.InProgress && r.status != Status.Requested) {
            revert WrongStatus(r.status);
        }
        if (modelBefore == bytes32(0) || modelAfter == bytes32(0)) revert ZeroHash();
        if (toRound < fromRound) revert EmptyRange();

        certId = ++certificateCount;
        certificates[certId] = Certificate({
            requestId: id, method: Method(method), fromRound: fromRound,
            toRound: toRound, modelBefore: modelBefore, modelAfter: modelAfter,
            samplesRemoved: samplesRemoved, metricBefore: metricBefore,
            metricAfter: metricAfter, durationSec: durationSec,
            issuedAt: uint64(block.timestamp), issuedBy: msg.sender,
            auditPassed: false, auditedBy: address(0)
        });
        r.status = Status.Certified;
        r.certificateId = certId;
        emit UnlearnCertified(id, certId, modelBefore, modelAfter,
                              metricBefore, metricAfter);
    }

    function fail(uint256 id, string calldata reason) external onlyService {
        Request storage r = requests[id];
        if (r.requestedAt == 0) revert UnknownRequest();
        r.status = Status.Failed;
        emit UnlearnFailed(id, reason);
    }

    /// @notice An auditor records that they independently recomputed the hashes
    /// and the metric, and whether it matched. Separate from the service, so
    /// the claim and its check never come from the same party.
    function audit(uint256 certId, bool passed, string calldata note) external {
        if (!registry.hasRole(msg.sender, AUDITOR)) revert NotAuditor();
        Certificate storage c = certificates[certId];
        if (c.issuedAt == 0) revert UnknownRequest();
        c.auditPassed = passed;
        c.auditedBy = msg.sender;
        emit CertificateAudited(certId, msg.sender, passed, note);
    }

    // ----------------------------------------------------------------- views

    function requestsFor(bytes32 subject) external view returns (uint256[] memory) {
        return _bySubject[subject];
    }

    function certificateOf(uint256 id) external view returns (Certificate memory) {
        return certificates[requests[id].certificateId];
    }

    /// @notice Everything an auditor needs, in one call.
    function auditBundle(uint256 certId)
        external
        view
        returns (bytes32 before_, bytes32 after_, uint256 fromRound,
                 uint256 toRound, uint8 method, uint32 metricBefore,
                 uint32 metricAfter)
    {
        Certificate storage c = certificates[certId];
        return (c.modelBefore, c.modelAfter, c.fromRound, c.toRound,
                uint8(c.method), c.metricBefore, c.metricAfter);
    }
}
