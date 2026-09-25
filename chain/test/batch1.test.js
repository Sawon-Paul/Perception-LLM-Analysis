const { expect } = require("chai");
const { ethers } = require("hardhat");

// Batch 1: Params, Registry, RoadPoint.
// Every test here maps to a rule the thesis claims the system enforces, so a
// failure means a security argument in the write-up is not actually true.

async function deployAll() {
  const [admin, v2, v3, v4, carA, carB, carC, merchant, faultLifecycle, outsider] =
    await ethers.getSigners();

  const Params = await ethers.getContractFactory("Params");
  const params = await Params.deploy([admin.address, v2.address, v3.address, v4.address]);

  const Registry = await ethers.getContractFactory("Registry");
  const registry = await Registry.deploy(admin.address, await params.getAddress());

  const RoadPoint = await ethers.getContractFactory("RoadPoint");
  const rp = await RoadPoint.deploy(admin.address, await registry.getAddress());

  await registry.setRoadPoint(await rp.getAddress());
  // FaultLifecycle is not written yet; a signer stands in for it so the
  // controller-only paths can be exercised.
  await rp.setController(faultLifecycle.address, true);
  await registry.setRole(merchant.address, await registry.MERCHANT(), true);

  return { admin, v2, v3, v4, carA, carB, carC, merchant, faultLifecycle,
           outsider, params, registry, rp };
}

const regHash = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

describe("Params — consortium governance", function () {
  it("ships the starting values from plan section 15", async function () {
    const { params } = await deployAll();
    expect(await params.get("k_confirm")).to.equal(3);
    expect(await params.get("N_rewarded")).to.equal(5);
    expect(await params.get("stake")).to.equal(5);
    expect(await params.get("registration_bond")).to.equal(25);
    expect(await params.get("T_pending")).to.equal(72 * 3600);
  });

  it("returns the reward table by position, 0 past the table", async function () {
    const { params } = await deployAll();
    expect(await params.rewardFor(1)).to.equal(10);
    expect(await params.rewardFor(2)).to.equal(6);
    expect(await params.rewardFor(3)).to.equal(4);
    expect(await params.rewardFor(4)).to.equal(2);
    expect(await params.rewardFor(5)).to.equal(1);
    expect(await params.rewardFor(6)).to.equal(0);
  });

  it("reverts on an unknown name instead of returning zero", async function () {
    // A silent zero for "stake" or "k_confirm" would disable a safety rule.
    const { params } = await deployAll();
    await expect(params.get("not_a_param")).to.be.revertedWithCustomError(
      params, "UnknownParam");
  });

  // plan 7.9: setParam fails with only 2 approvals
  it("does not change a value on two approvals", async function () {
    const { params, admin, v2 } = await deployAll();
    await params.connect(admin).approveParam("k_confirm", 4);
    await params.connect(v2).approveParam("k_confirm", 4);
    expect(await params.get("k_confirm")).to.equal(3);
  });

  it("changes it on the third", async function () {
    const { params, admin, v2, v3 } = await deployAll();
    await params.connect(admin).approveParam("k_confirm", 4);
    await params.connect(v2).approveParam("k_confirm", 4);
    await expect(params.connect(v3).approveParam("k_confirm", 4))
      .to.emit(params, "ParamChanged").withArgs("k_confirm", 3, 4);
    expect(await params.get("k_confirm")).to.equal(4);
  });

  it("stops one member approving three times", async function () {
    const { params, admin } = await deployAll();
    await params.connect(admin).approveParam("stake", 50);
    await expect(params.connect(admin).approveParam("stake", 50))
      .to.be.revertedWithCustomError(params, "AlreadyApproved");
    expect(await params.get("stake")).to.equal(5);
  });

  it("rejects a non-member", async function () {
    const { params, outsider } = await deployAll();
    await expect(params.connect(outsider).approveParam("stake", 1))
      .to.be.revertedWithCustomError(params, "NotConsortium");
  });

  it("lets an approval be withdrawn before the threshold", async function () {
    const { params, admin, v2, v3 } = await deployAll();
    await params.connect(admin).approveParam("stake", 9);
    await params.connect(v2).approveParam("stake", 9);
    await params.connect(v2).revokeApproval("stake", 9);
    await params.connect(v3).approveParam("stake", 9);
    expect(await params.get("stake")).to.equal(5);
  });
});

describe("Registry — identity and Sybil resistance", function () {
  it("registers a car and mints its bond", async function () {
    const { registry, rp, carA, admin } = await deployAll();
    await expect(registry.connect(admin).registerCar(
      carA.address, regHash("owner1"), regHash("vehicle1")))
      .to.emit(registry, "CarRegistered");
    expect(await registry.isActive(carA.address)).to.equal(true);
    expect(await rp.balanceOf(carA.address)).to.equal(25);
    expect(await rp.bondOf(carA.address)).to.equal(25);
  });

  // plan 13.1: Sybil attack — try to register the same vehicle twice
  it("refuses a second registration of the same vehicle", async function () {
    const { registry, carA, carB, admin } = await deployAll();
    await registry.connect(admin).registerCar(
      carA.address, regHash("owner1"), regHash("vehicle1"));
    await expect(registry.connect(admin).registerCar(
      carB.address, regHash("owner2"), regHash("vehicle1")))
      .to.be.revertedWithCustomError(registry, "VehicleAlreadyRegistered");
  });

  // plan 7.9: same owner with two cars cannot count twice toward k
  it("reports two cars of one owner as the same owner", async function () {
    const { registry, carA, carB, carC, admin } = await deployAll();
    await registry.connect(admin).registerCar(
      carA.address, regHash("owner1"), regHash("v1"));
    await registry.connect(admin).registerCar(
      carB.address, regHash("owner1"), regHash("v2"));
    await registry.connect(admin).registerCar(
      carC.address, regHash("owner2"), regHash("v3"));
    expect(await registry.sameOwner(carA.address, carB.address)).to.equal(true);
    expect(await registry.sameOwner(carA.address, carC.address)).to.equal(false);
  });

  it("does not treat two unregistered addresses as sharing an owner", async function () {
    const { registry, outsider, merchant } = await deployAll();
    expect(await registry.sameOwner(outsider.address, merchant.address))
      .to.equal(false);
  });

  it("emits CarLeft on deregistration, which triggers unlearning", async function () {
    const { registry, carA, admin } = await deployAll();
    await registry.connect(admin).registerCar(
      carA.address, regHash("owner1"), regHash("v1"));
    await expect(registry.connect(carA).deregisterCar(carA.address, "EXIT"))
      .to.emit(registry, "CarLeft");
    expect(await registry.isActive(carA.address)).to.equal(false);
  });

  it("keeps the vehicle hash claimed after leaving", async function () {
    // Otherwise an owner could deregister and re-register to shed a bad
    // reputation, which would defeat the whole penalty system.
    const { registry, carA, carB, admin } = await deployAll();
    await registry.connect(admin).registerCar(
      carA.address, regHash("owner1"), regHash("v1"));
    await registry.connect(admin).deregisterCar(carA.address, "EXIT");
    await expect(registry.connect(admin).registerCar(
      carB.address, regHash("owner1"), regHash("v1")))
      .to.be.revertedWithCustomError(registry, "VehicleAlreadyRegistered");
  });

  it("lets only the admin register", async function () {
    const { registry, carA, outsider } = await deployAll();
    await expect(registry.connect(outsider).registerCar(
      carA.address, regHash("o"), regHash("v")))
      .to.be.revertedWithCustomError(registry, "NotAdmin");
  });

  it("sets roles and reads them back", async function () {
    const { registry, admin, outsider } = await deployAll();
    const KEEPER = await registry.KEEPER();
    expect(await registry.hasRole(outsider.address, KEEPER)).to.equal(false);
    await registry.connect(admin).setRole(outsider.address, KEEPER, true);
    expect(await registry.hasRole(outsider.address, KEEPER)).to.equal(true);
  });

  it("lets the unlearning service exclude a car from FL", async function () {
    const { registry, admin, carA, outsider } = await deployAll();
    await registry.connect(admin).registerCar(
      carA.address, regHash("o1"), regHash("v1"));
    await registry.connect(admin).setRole(
      outsider.address, await registry.UNLEARN_SERVICE(), true);
    await registry.connect(outsider).setFLExclusion(carA.address, true);
    expect(await registry.isExcludedFromFL(carA.address)).to.equal(true);
  });
});

describe("RoadPoint — a reward point, not a currency", function () {
  async function withCar() {
    const ctx = await deployAll();
    await ctx.registry.connect(ctx.admin).registerCar(
      ctx.carA.address, regHash("owner1"), regHash("v1"));
    await ctx.registry.connect(ctx.admin).registerCar(
      ctx.carB.address, regHash("owner2"), regHash("v2"));
    return ctx;
  }

  // plan 7.9: car-to-car transfer reverts
  it("reverts on transfer between cars", async function () {
    const { rp, carA, carB } = await withCar();
    await expect(rp.connect(carA).transfer(carB.address, 1))
      .to.be.revertedWithCustomError(rp, "TransfersDisabled");
  });

  it("reverts on approve and transferFrom", async function () {
    const { rp, carA, carB } = await withCar();
    await expect(rp.connect(carA).approve(carB.address, 1))
      .to.be.revertedWithCustomError(rp, "ApprovalsDisabled");
    await expect(rp.connect(carB).transferFrom(carA.address, carB.address, 1))
      .to.be.revertedWithCustomError(rp, "TransfersDisabled");
  });

  it("keeps the bond out of the spendable balance", async function () {
    const { rp, carA } = await withCar();
    expect(await rp.balanceOf(carA.address)).to.equal(25);
    expect(await rp.freeBalance(carA.address)).to.equal(0);
    expect(await rp.stakeable(carA.address)).to.equal(25);
  });

  it("locks, unlocks and slashes a stake", async function () {
    const { rp, carA, faultLifecycle } = await withCar();
    await rp.connect(faultLifecycle).lock(carA.address, 5);
    expect(await rp.lockedOf(carA.address)).to.equal(5);
    expect(await rp.stakeable(carA.address)).to.equal(20);

    await rp.connect(faultLifecycle).unlock(carA.address, 5);
    expect(await rp.lockedOf(carA.address)).to.equal(0);

    await rp.connect(faultLifecycle).lock(carA.address, 5);
    await expect(rp.connect(faultLifecycle).slash(carA.address, 5))
      .to.emit(rp, "Slashed").withArgs(carA.address, 5);
    expect(await rp.balanceOf(carA.address)).to.equal(20);
    expect(await rp.bondOf(carA.address)).to.equal(20);
  });

  it("refuses to lock more than the car holds", async function () {
    const { rp, carA, faultLifecycle } = await withCar();
    await expect(rp.connect(faultLifecycle).lock(carA.address, 999))
      .to.be.revertedWithCustomError(rp, "InsufficientFree");
  });

  it("lets only a controller mint, lock or slash", async function () {
    const { rp, carA, outsider } = await withCar();
    for (const call of [
      rp.connect(outsider).mint(carA.address, 10),
      rp.connect(outsider).lock(carA.address, 1),
      rp.connect(outsider).slash(carA.address, 1),
    ]) {
      await expect(call).to.be.revertedWithCustomError(rp, "NotController");
    }
  });

  it("redeems at a merchant with the car's signature", async function () {
    const { rp, carA, merchant, faultLifecycle } = await withCar();
    await rp.connect(faultLifecycle).mint(carA.address, 10);
    expect(await rp.freeBalance(carA.address)).to.equal(10);

    const merchantId = regHash("toll-gate-1");
    const nonce = await rp.nonceOf(carA.address);
    const digest = await rp.redeemDigest(carA.address, 6, merchantId, nonce);
    const sig = await carA.signMessage(ethers.getBytes(
      ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
        ["uint256", "address", "address", "uint256", "bytes32", "uint256"],
        [(await ethers.provider.getNetwork()).chainId, await rp.getAddress(),
         carA.address, 6, merchantId, nonce]))));

    await expect(rp.connect(merchant).redeem(carA.address, 6, merchantId, sig))
      .to.emit(rp, "Redeemed").withArgs(carA.address, merchantId, 6);
    expect(await rp.freeBalance(carA.address)).to.equal(4);
    expect(digest).to.be.a("string");
  });

  // plan 7.9: redeem by a non-merchant reverts
  it("refuses redemption by a non-merchant", async function () {
    const { rp, carA, outsider, faultLifecycle } = await withCar();
    await rp.connect(faultLifecycle).mint(carA.address, 10);
    await expect(rp.connect(outsider).redeem(
      carA.address, 1, regHash("m"), "0x00"))
      .to.be.revertedWithCustomError(rp, "NotMerchant");
  });

  // plan 13.1: double redeem — the same tokens at two merchants
  it("refuses a replayed redemption signature", async function () {
    const { rp, carA, merchant, faultLifecycle } = await withCar();
    await rp.connect(faultLifecycle).mint(carA.address, 20);

    const merchantId = regHash("fuel-1");
    const nonce = await rp.nonceOf(carA.address);
    const chainId = (await ethers.provider.getNetwork()).chainId;
    const sig = await carA.signMessage(ethers.getBytes(
      ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
        ["uint256", "address", "address", "uint256", "bytes32", "uint256"],
        [chainId, await rp.getAddress(), carA.address, 5, merchantId, nonce]))));

    await rp.connect(merchant).redeem(carA.address, 5, merchantId, sig);
    await expect(rp.connect(merchant).redeem(carA.address, 5, merchantId, sig))
      .to.be.revertedWithCustomError(rp, "BadSignature");
  });

  it("refuses redemption of the bond", async function () {
    const { rp, carA, merchant } = await withCar();
    // free balance is 0 while only the bond is held
    const merchantId = regHash("m");
    const nonce = await rp.nonceOf(carA.address);
    const chainId = (await ethers.provider.getNetwork()).chainId;
    const sig = await carA.signMessage(ethers.getBytes(
      ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
        ["uint256", "address", "address", "uint256", "bytes32", "uint256"],
        [chainId, await rp.getAddress(), carA.address, 10, merchantId, nonce]))));
    await expect(rp.connect(merchant).redeem(carA.address, 10, merchantId, sig))
      .to.be.revertedWithCustomError(rp, "InsufficientFree");
  });

  it("claws back rewards, recording a debt when the balance is short", async function () {
    const { rp, carA, faultLifecycle } = await withCar();
    await rp.connect(faultLifecycle).mint(carA.address, 10);
    await rp.connect(faultLifecycle).clawback(carA.address, 16);
    expect(await rp.freeBalance(carA.address)).to.equal(0);
    expect(await rp.debtOf(carA.address)).to.equal(6);

    // future rewards pay the debt before the car sees anything
    await rp.connect(faultLifecycle).mint(carA.address, 10);
    expect(await rp.debtOf(carA.address)).to.equal(0);
    expect(await rp.freeBalance(carA.address)).to.equal(4);
  });

  it("does not let a clawback eat the bond", async function () {
    const { rp, carA, faultLifecycle } = await withCar();
    await rp.connect(faultLifecycle).clawback(carA.address, 20);
    expect(await rp.bondOf(carA.address)).to.equal(25);
    expect(await rp.balanceOf(carA.address)).to.equal(25);
    expect(await rp.debtOf(carA.address)).to.equal(20);
  });
});
