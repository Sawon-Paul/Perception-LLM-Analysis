// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Registry
/// @notice Who is a car, who owns it, and who holds a special role.
///
/// Plan section 7.3. Two rules here carry most of the security argument.
///
/// One registration per vehicle registration hash. This is the Sybil defence:
/// confirmation requires k *distinct owners*, so an attacker who could mint
/// identities freely could confirm any fake fault alone. The vehicle document
/// is the scarce thing an attacker cannot copy.
///
/// Owner identity is stored as a hash, never a name or a document. The chain is
/// publicly readable, so putting an owner's identity on it would publish who
/// drives where.
interface IRoadPoint {
    function mintBond(address car, uint256 amount) external;
}

interface IParams {
    function get(string calldata name) external view returns (uint256);
}

contract Registry {
    bytes32 public constant REPAIR = keccak256("REPAIR");
    bytes32 public constant MERCHANT = keccak256("MERCHANT");
    bytes32 public constant FL_SERVER = keccak256("FL_SERVER");
    bytes32 public constant UNLEARN_SERVICE = keccak256("UNLEARN_SERVICE");
    bytes32 public constant KEEPER = keccak256("KEEPER");
    bytes32 public constant AUDITOR = keccak256("AUDITOR");

    struct Car {
        bytes32 ownerId;        // hash of the owner's identity, never the identity
        bytes32 vehicleRegHash; // hash of the vehicle registration document
        bool active;
        bool excludedFromFL;    // set after client unlearning
        uint64 registeredAt;
    }

    address public admin;               // road authority
    IParams public params;
    IRoadPoint public roadPoint;

    mapping(address => Car) private _cars;
    mapping(bytes32 => address) public carByVehicleHash;
    mapping(address => mapping(bytes32 => bool)) private _roles;
    uint256 public carCount;

    event CarRegistered(address indexed car, bytes32 indexed ownerId, uint64 at);
    event CarLeft(address indexed car, bytes32 indexed ownerId, string reason);
    event RoleSet(address indexed who, bytes32 indexed role, bool enabled);
    event FLExclusionSet(address indexed car, bool excluded);
    event AdminChanged(address oldAdmin, address newAdmin);

    error NotAdmin();
    error VehicleAlreadyRegistered(address existingCar);
    error UnknownCar();
    error ZeroAddress();

    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAdmin();
        _;
    }

    constructor(address admin_, address params_) {
        if (admin_ == address(0) || params_ == address(0)) revert ZeroAddress();
        admin = admin_;
        params = IParams(params_);
    }

    /// Set once after RoadPoint is deployed, since the two reference each other.
    function setRoadPoint(address roadPoint_) external onlyAdmin {
        if (roadPoint_ == address(0)) revert ZeroAddress();
        roadPoint = IRoadPoint(roadPoint_);
    }

    function setAdmin(address newAdmin) external onlyAdmin {
        if (newAdmin == address(0)) revert ZeroAddress();
        emit AdminChanged(admin, newAdmin);
        admin = newAdmin;
    }

    /// @notice Register a car and mint its locked registration bond.
    /// @param vehicleRegHash keccak256 of the vehicle registration document.
    /// Reverts if that document is already registered, which is what stops one
    /// attacker from becoming k distinct owners.
    function registerCar(address car, bytes32 ownerId, bytes32 vehicleRegHash)
        external
        onlyAdmin
    {
        if (car == address(0)) revert ZeroAddress();
        address existing = carByVehicleHash[vehicleRegHash];
        if (existing != address(0)) revert VehicleAlreadyRegistered(existing);

        _cars[car] = Car({
            ownerId: ownerId,
            vehicleRegHash: vehicleRegHash,
            active: true,
            excludedFromFL: false,
            registeredAt: uint64(block.timestamp)
        });
        carByVehicleHash[vehicleRegHash] = car;
        carCount++;

        emit CarRegistered(car, ownerId, uint64(block.timestamp));

        if (address(roadPoint) != address(0)) {
            roadPoint.mintBond(car, params.get("registration_bond"));
        }
    }

    /// @notice Retire a car. Emits CarLeft, which the unlearning service treats
    /// as a request to remove that car's contribution from the global model.
    function deregisterCar(address car, string calldata reason) external {
        Car storage c = _cars[car];
        if (c.registeredAt == 0) revert UnknownCar();
        if (msg.sender != admin && msg.sender != car) revert NotAdmin();

        c.active = false;
        // The vehicle hash stays claimed: freeing it would let an owner
        // deregister and re-register to shed a bad reputation.
        emit CarLeft(car, c.ownerId, reason);
    }

    function setRole(address who, bytes32 role, bool enabled) external onlyAdmin {
        if (who == address(0)) revert ZeroAddress();
        _roles[who][role] = enabled;
        emit RoleSet(who, role, enabled);
    }

    /// Called by the unlearning service once a car's influence is removed, so
    /// the FL server skips its future updates.
    function setFLExclusion(address car, bool excluded) external {
        if (msg.sender != admin && !_roles[msg.sender][UNLEARN_SERVICE]) {
            revert NotAdmin();
        }
        if (_cars[car].registeredAt == 0) revert UnknownCar();
        _cars[car].excludedFromFL = excluded;
        emit FLExclusionSet(car, excluded);
    }

    // ------------------------------------------------------------------ views

    function ownerOf(address car) external view returns (bytes32) {
        return _cars[car].ownerId;
    }

    function isActive(address car) external view returns (bool) {
        return _cars[car].active;
    }

    function isRegistered(address car) external view returns (bool) {
        return _cars[car].registeredAt != 0;
    }

    function isExcludedFromFL(address car) external view returns (bool) {
        return _cars[car].excludedFromFL;
    }

    function hasRole(address who, bytes32 role) external view returns (bool) {
        return _roles[who][role];
    }

    function carInfo(address car) external view returns (Car memory) {
        return _cars[car];
    }

    /// @notice Do these two cars share an owner? The distinct-owner rule in
    /// FaultLifecycle calls this, and it is what blocks the collusion attack
    /// where one person runs three cars.
    function sameOwner(address a, address b) external view returns (bool) {
        bytes32 oa = _cars[a].ownerId;
        return oa != bytes32(0) && oa == _cars[b].ownerId;
    }
}
