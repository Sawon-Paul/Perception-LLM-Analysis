// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";

/// @title RoadPoint
/// @notice A reward point that is deliberately not a cryptocurrency.
///
/// Plan section 7.4. Cryptocurrency trading is illegal in Bangladesh, so this
/// token must never behave like one. Every path that would make it tradable is
/// closed: transfer and transferFrom revert, approve reverts, and the only way
/// out is redemption at a registered toll or fuel merchant, which burns the
/// points. There is no market, no exchange rate on-chain, and no way to move
/// value between two cars.
///
/// It keeps the ERC-20 interface only so that balances and events are readable
/// by standard tooling.
///
/// Balances are split in two. Bond is the locked amount minted at registration:
/// it backs stakes and can never be redeemed, so it is not a reward and cannot
/// be cashed out. Earned points are what a car actually gets to spend.
interface IRegistryView {
    function isActive(address car) external view returns (bool);
    function hasRole(address who, bytes32 role) external view returns (bool);
}

contract RoadPoint is ERC20 {
    using ECDSA for bytes32;

    bytes32 public constant MERCHANT = keccak256("MERCHANT");

    address public admin;
    IRegistryView public registry;

    /// contracts allowed to mint, lock, slash — set to FaultLifecycle
    mapping(address => bool) public isController;

    /// locked registration bond, never redeemable
    mapping(address => uint256) public bondOf;
    /// amount currently locked as stake on open reports
    mapping(address => uint256) public lockedOf;
    /// owed back after a clawback found the balance too small
    mapping(address => uint256) public debtOf;
    /// replay protection for redemptions
    mapping(address => uint256) public nonceOf;

    event ControllerSet(address indexed who, bool enabled);
    event BondMinted(address indexed car, uint256 amount);
    event Locked(address indexed car, uint256 amount);
    event Unlocked(address indexed car, uint256 amount);
    event Slashed(address indexed car, uint256 amount);
    event Redeemed(address indexed car, bytes32 indexed merchantId, uint256 amount);
    event ClawedBack(address indexed car, uint256 burned, uint256 debtAdded);
    event DebtRepaid(address indexed car, uint256 amount, uint256 remaining);

    error NotAdmin();
    error NotController();
    error NotMerchant();
    error TransfersDisabled();
    error ApprovalsDisabled();
    error InsufficientFree(uint256 free, uint256 needed);
    error InsufficientLocked(uint256 locked, uint256 needed);
    error BondNotRedeemable();
    error BadSignature();
    error CarNotActive();

    modifier onlyAdmin() {
        if (msg.sender != admin) revert NotAdmin();
        _;
    }
    modifier onlyController() {
        if (!isController[msg.sender]) revert NotController();
        _;
    }

    constructor(address admin_, address registry_) ERC20("RoadPoint", "RP") {
        admin = admin_;
        registry = IRegistryView(registry_);
    }

    function setController(address who, bool enabled) external onlyAdmin {
        isController[who] = enabled;
        emit ControllerSet(who, enabled);
    }

    function setRegistry(address registry_) external onlyAdmin {
        registry = IRegistryView(registry_);
    }

    // ----------------------------------------------- the non-transferable part

    /// Car-to-car transfer is the thing that would make this a currency.
    function transfer(address, uint256) public pure override returns (bool) {
        revert TransfersDisabled();
    }

    function transferFrom(address, address, uint256) public pure override returns (bool) {
        revert TransfersDisabled();
    }

    /// Approvals exist only to enable transferFrom, so they are closed too.
    function approve(address, uint256) public pure override returns (bool) {
        revert ApprovalsDisabled();
    }

    // -------------------------------------------------------------- accounting

    /// Points a car may actually spend: balance minus bond minus locked stake.
    function freeBalance(address car) public view returns (uint256) {
        uint256 bal = balanceOf(car);
        uint256 reserved = bondOf[car] + lockedOf[car];
        return bal > reserved ? bal - reserved : 0;
    }

    /// Bond plus earned points that are not currently staked.
    function stakeable(address car) public view returns (uint256) {
        uint256 bal = balanceOf(car);
        return bal > lockedOf[car] ? bal - lockedOf[car] : 0;
    }

    function mintBond(address car, uint256 amount) external {
        if (msg.sender != admin && !isController[msg.sender]
            && msg.sender != address(registry)) revert NotController();
        bondOf[car] += amount;
        _mint(car, amount);
        emit BondMinted(car, amount);
    }

    /// Reward for a confirmed report. Any outstanding debt is paid first, so a
    /// car proven fraudulent cannot simply keep earning past its clawback.
    function mint(address car, uint256 amount) external onlyController {
        uint256 debt = debtOf[car];
        if (debt > 0) {
            uint256 pay = debt < amount ? debt : amount;
            debtOf[car] = debt - pay;
            amount -= pay;
            emit DebtRepaid(car, pay, debtOf[car]);
        }
        if (amount > 0) _mint(car, amount);
    }

    function lock(address car, uint256 amount) external onlyController {
        uint256 free = stakeable(car);
        if (free < amount) revert InsufficientFree(free, amount);
        lockedOf[car] += amount;
        emit Locked(car, amount);
    }

    function unlock(address car, uint256 amount) external onlyController {
        uint256 locked = lockedOf[car];
        if (locked < amount) revert InsufficientLocked(locked, amount);
        lockedOf[car] = locked - amount;
        emit Unlocked(car, amount);
    }

    /// Burn a locked stake after a fault is rejected. Burns bond first, because
    /// the bond is what the stake was drawn against.
    function slash(address car, uint256 amount) external onlyController {
        uint256 locked = lockedOf[car];
        if (locked < amount) revert InsufficientLocked(locked, amount);
        lockedOf[car] = locked - amount;

        uint256 bond = bondOf[car];
        uint256 fromBond = bond < amount ? bond : amount;
        bondOf[car] = bond - fromBond;

        _burn(car, amount);
        emit Slashed(car, amount);
    }

    /// Remove rewards already paid for a fault later proven fake. If the car
    /// has spent them, the shortfall becomes a debt that future rewards repay.
    function clawback(address car, uint256 amount) external onlyController {
        uint256 free = freeBalance(car);
        uint256 burn = free < amount ? free : amount;
        uint256 debt = amount - burn;
        if (burn > 0) _burn(car, burn);
        if (debt > 0) debtOf[car] += debt;
        emit ClawedBack(car, burn, debt);
    }

    // -------------------------------------------------------------- redemption

    /// Message the car signs to authorise a redemption at a merchant.
    function redeemDigest(address car, uint256 amount, bytes32 merchantId, uint256 nonce)
        public
        view
        returns (bytes32)
    {
        return MessageHashUtils.toEthSignedMessageHash(
            keccak256(abi.encode(block.chainid, address(this), car, amount, merchantId, nonce))
        );
    }

    /// @notice Burn points in exchange for toll or fuel. Called by the merchant
    /// with a signature from the car, so the car must consent and the merchant
    /// cannot drain a balance on its own.
    ///
    /// The nonce is what stops the double-redeem attack: a replayed signature
    /// carries a stale nonce and reverts. With QBFT finality the merchant can
    /// accept as soon as the transaction is mined.
    function redeem(
        address car,
        uint256 amount,
        bytes32 merchantId,
        bytes calldata signature
    ) external {
        if (!registry.hasRole(msg.sender, MERCHANT)) revert NotMerchant();
        if (!registry.isActive(car)) revert CarNotActive();

        uint256 free = freeBalance(car);
        if (free < amount) revert InsufficientFree(free, amount);

        uint256 nonce = nonceOf[car];
        bytes32 digest = redeemDigest(car, amount, merchantId, nonce);
        if (digest.recover(signature) != car) revert BadSignature();

        nonceOf[car] = nonce + 1;
        _burn(car, amount);
        emit Redeemed(car, merchantId, amount);
    }

    /// Explicit rather than implicit: bond can never leave as value.
    function redeemBond(address, uint256) external pure {
        revert BondNotRedeemable();
    }
}
