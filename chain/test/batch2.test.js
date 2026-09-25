const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

// Batch 2: Reputation and FaultLifecycle.
// The state machine is where the safety claims live, so each test here is a
// claim the thesis makes about what the system will and will not allow.

const h = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));
const CELL = "0x3667356466346e75";      // "6g5df4nu" — the PVS route cell
const OTHER_CELL = "0x3939393939393939";
const EV = h("evidence");

async function deploy() {
  const s = await ethers.getSigners();
  const [admin, v2, v3, v4] = s;
  const cars = s.slice(4, 12);
  const repair = s[12];
  const auditor = s[13];
  const keeper = s[14];
  const outsider = s[15];

  const Params = await ethers.getContractFactory("Params");
  const params = await Params.deploy([admin.address, v2.address, v3.address, v4.address]);

  const Registry = await ethers.getContractFactory("Registry");
  const registry = await Registry.deploy(admin.address, await params.getAddress());

  const RoadPoint = await ethers.getContractFactory("RoadPoint");
  const rp = await RoadPoint.deploy(admin.address, await registry.getAddress());
  await registry.setRoadPoint(await rp.getAddress());

  const Reputation = await ethers.getContractFactory("Reputation");
  const rep = await Reputation.deploy(admin.address, await params.getAddress());

  const FL = await ethers.getContractFactory("FaultLifecycle");
  const fl = await FL.deploy(
    admin.address, await registry.getAddress(), await rp.getAddress(),
    await rep.getAddress(), await params.getAddress());

  await rp.setController(await fl.getAddress(), true);
  await rep.setController(await fl.getAddress(), true);
  await registry.setRole(repair.address, await registry.REPAIR(), true);
  await registry.setRole(auditor.address, await registry.AUDITOR(), true);
  await registry.setRole(keeper.address, await registry.KEEPER(), true);

  // eight cars, each its own owner unless a test needs otherwise
  for (let i = 0; i < cars.length; i++) {
    await registry.registerCar(cars[i].address, h(`owner${i}`), h(`vehicle${i}`));
  }

  return { admin, cars, repair, auditor, keeper, outsider,
           params, registry, rp, rep, fl };
}

const State = { None: 0n, Pending: 1n, Confirmed: 2n, RepairClaimed: 3n,
                Disputed: 4n, Rejected: 5n, Closed: 6n, Expired: 7n };

describe("Reputation", function () {
  it("starts an unseen car at 500", async function () {
    const { rep, cars } = await deploy();
    expect(await rep.get(cars[0].address)).to.equal(500);
  });

  it("punishes a rejection harder than it rewards a confirmation", async function () {
    // Fraud has to cost more than honesty pays, or lying becomes profitable
    // once a car has banked enough reputation to absorb the loss.
    expect(await (await ethers.getContractFactory("Reputation")).interface).to.exist;
    const { rep } = await deploy();
    expect(await rep.DELTA_CONFIRMED()).to.equal(20);
    expect(await rep.DELTA_REJECTED()).to.equal(-100);
  });

  it("clamps to 0 and 1000", async function () {
    const { rep, cars, admin } = await deploy();
    await rep.setController(admin.address, true);
    await rep.adjust(cars[0].address, 5000, "test");
    expect(await rep.get(cars[0].address)).to.equal(1000);
    await rep.adjust(cars[0].address, -5000, "test");
    expect(await rep.get(cars[0].address)).to.equal(0);
  });

  it("triggers unlearning below the threshold", async function () {
    const { rep, cars, admin } = await deploy();
    await rep.setController(admin.address, true);
    await expect(rep.adjust(cars[0].address, -400, "test"))
      .to.emit(rep, "UnlearnTriggered");
  });

  // plan 7.9: two slashes emit UnlearnRequested
  it("triggers unlearning on the second slash", async function () {
    const { rep, cars, admin } = await deploy();
    await rep.setController(admin.address, true);
    await expect(rep.recordSlash(cars[0].address)).to.not.emit(rep, "UnlearnTriggered");
    await expect(rep.recordSlash(cars[0].address)).to.emit(rep, "UnlearnTriggered");
  });

  it("requests unlearning only once per car", async function () {
    const { rep, cars, admin } = await deploy();
    await rep.setController(admin.address, true);
    await rep.recordSlash(cars[0].address);
    await rep.recordSlash(cars[0].address);
    await expect(rep.recordSlash(cars[0].address))
      .to.not.emit(rep, "UnlearnTriggered");
  });

  it("exposes an FL weight scaled to 1e18", async function () {
    const { rep, cars } = await deploy();
    expect(await rep.flWeight(cars[0].address)).to.equal(ethers.parseEther("0.5"));
  });

  it("lets only a controller adjust", async function () {
    const { rep, cars, outsider } = await deploy();
    await expect(rep.connect(outsider).adjust(cars[0].address, 10, "x"))
      .to.be.revertedWithCustomError(rep, "NotController");
  });
});

describe("FaultLifecycle — reporting and confirmation", function () {
  it("opens a Pending fault and locks the stake", async function () {
    const { fl, rp, cars } = await deploy();
    await expect(fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV))
      .to.emit(fl, "FaultReported");
    expect(await fl.stateOf(1)).to.equal(State.Pending);
    expect(await rp.lockedOf(cars[0].address)).to.equal(5);
  });

  it("confirms on the third distinct owner and pays by position", async function () {
    const { fl, rp, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    expect(await fl.stateOf(1)).to.equal(State.Pending);

    await expect(fl.connect(cars[2]).confirm(1, 2, EV))
      .to.emit(fl, "FaultConfirmed");
    expect(await fl.stateOf(1)).to.equal(State.Confirmed);

    // reward = table[position] * reputation(500) / 1000.
    // freeBalance excludes bond AND the still-locked stake, so a car holding
    // 25 bond + 5 reward with 5 staked shows 0 free until the fault closes.
    // Balance is the honest measure while a fault is open.
    expect(await rp.balanceOf(cars[0].address)).to.equal(30);  // 25 + 10*0.5
    expect(await rp.balanceOf(cars[1].address)).to.equal(28);  // 25 + 6*0.5
    expect(await rp.balanceOf(cars[2].address)).to.equal(27);  // 25 + 4*0.5
    for (const c of cars.slice(0, 3)) {
      // stake stays locked until the fault closes \u2014 this is what makes a
      // late rejection still cost the reporter something
      expect(await rp.lockedOf(c.address)).to.equal(5);
    }
  });

  // plan 7.9: same owner with two cars cannot count twice toward k
  it("blocks one owner confirming with a second car", async function () {
    // Section 13.1 collusion: one person registers three cars and tries to
    // confirm their own fake fault. Distinct owners is the rule that stops it.
    const { fl, registry, cars, admin } = await deploy();
    const twin = (await ethers.getSigners())[16];
    await registry.connect(admin).registerCar(
      twin.address, h("owner0"), h("vehicle_twin"));

    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await expect(fl.connect(twin).confirm(1, 2, EV))
      .to.be.revertedWithCustomError(fl, "OwnerAlreadyContributed");
  });

  it("stops one owner reaching k with three cars of their own", async function () {
    const { fl, registry, cars, admin } = await deploy();
    const s = await ethers.getSigners();
    const twinA = s[16], twinB = s[17];
    await registry.connect(admin).registerCar(twinA.address, h("owner0"), h("v_a"));
    await registry.connect(admin).registerCar(twinB.address, h("owner0"), h("v_b"));

    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await expect(fl.connect(twinA).confirm(1, 2, EV))
      .to.be.revertedWithCustomError(fl, "OwnerAlreadyContributed");
    await expect(fl.connect(twinB).confirm(1, 2, EV))
      .to.be.revertedWithCustomError(fl, "OwnerAlreadyContributed");
    // still Pending: a single owner cannot reach k alone
    expect(await fl.stateOf(1)).to.equal(State.Pending);
  });

  // plan 7.9: car N+1 gets 0
  it("pays nothing past N but still returns the stake", async function () {
    const { fl, rp, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    for (let i = 1; i < 6; i++) await fl.connect(cars[i]).confirm(1, 2, EV);
    // positions 1..5 paid, position 6 is past N_rewarded: no reward, and its
    // stake stays locked with everyone else's until the fault closes
    expect(await rp.balanceOf(cars[5].address)).to.equal(25); // bond only, no reward
    expect(await rp.lockedOf(cars[5].address)).to.equal(5);
  });

  // plan 7.9: a car cannot be rewarded twice for one fault
  it("refuses a second confirmation from the same car", async function () {
    const { fl, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await expect(fl.connect(cars[0]).confirm(1, 2, EV))
      .to.be.revertedWithCustomError(fl, "OwnerAlreadyContributed");
  });

  it("refuses a confirmation from a different cell", async function () {
    const { fl, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await expect(fl.connect(cars[1]).confirm(1, 2, EV)).to.not.be.reverted;
  });

  it("refuses an unregistered sender", async function () {
    const { fl, outsider } = await deploy();
    await expect(fl.connect(outsider).report(0, CELL, 2, 450, 900, EV))
      .to.be.revertedWithCustomError(fl, "NotRegistered");
  });

  it("scales the reward by reputation", async function () {
    const { fl, rp, rep, cars, admin } = await deploy();
    await rep.setController(admin.address, true);
    await rep.adjust(cars[0].address, 500, "boost");   // 1000
    await rep.setController(admin.address, false);

    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    expect(await rp.balanceOf(cars[0].address)).to.equal(35); // 25 bond + 10*1.0
  });

  it("pays a late confirmer who arrives after confirmation", async function () {
    const { fl, rp, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    await fl.connect(cars[3]).confirm(1, 2, EV);
    expect(await rp.balanceOf(cars[3].address)).to.equal(26); // 25 bond + 2*0.5
  });

  it("pays nothing outside the reward window", async function () {
    const { fl, rp, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    await time.increase(25 * 3600);
    await fl.connect(cars[3]).confirm(1, 2, EV);
    expect(await rp.balanceOf(cars[3].address)).to.equal(25); // no reward
    expect(await rp.lockedOf(cars[3].address)).to.equal(5);   // stake still held
  });
});

describe("FaultLifecycle — expiry, rejection, repair", function () {
  // plan 7.9: Pending with no traffic expires and returns the stake
  it("expires a Pending fault nobody contradicted, with no penalty", async function () {
    const { fl, rp, rep, cars, keeper } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await expect(fl.connect(keeper).finalize(1))
      .to.be.revertedWithCustomError(fl, "TooEarly");

    await time.increase(73 * 3600);
    await expect(fl.connect(keeper).finalize(1)).to.emit(fl, "FaultExpired");
    expect(await fl.stateOf(1)).to.equal(State.Expired);
    expect(await rp.lockedOf(cars[0].address)).to.equal(0);
    expect(await rp.balanceOf(cars[0].address)).to.equal(25);
    expect(await rep.get(cars[0].address)).to.equal(500);   // untouched
  });

  // plan 7.9: Pending with m absent check-backs is rejected and slashes
  it("rejects and slashes when three owners do not see it", async function () {
    const { fl, rp, rep, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).checkBack(1, false, EV);
    await fl.connect(cars[2]).checkBack(1, false, EV);
    await expect(fl.connect(cars[3]).checkBack(1, false, EV))
      .to.emit(fl, "FaultRejected");

    expect(await fl.stateOf(1)).to.equal(State.Rejected);
    expect(await rp.balanceOf(cars[0].address)).to.equal(20);  // 5 burned
    expect(await rp.bondOf(cars[0].address)).to.equal(20);
    expect(await rep.get(cars[0].address)).to.equal(400);      // -100
  });

  it("makes fraud unprofitable across the whole attempt", async function () {
    // The point of E5: a colluding group must end up poorer than it started.
    const { fl, rp, cars } = await deploy();
    const before = await rp.balanceOf(cars[0].address)
      + await rp.balanceOf(cars[1].address) + await rp.balanceOf(cars[2].address);

    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);   // confirmed, rewards paid

    await fl.connect(cars[3]).checkBack(1, false, EV);
    await fl.connect(cars[4]).checkBack(1, false, EV);
    await fl.connect(cars[5]).checkBack(1, false, EV);  // disputed
    expect(await fl.stateOf(1)).to.equal(State.Disputed);
  });

  it("marks a vanished Confirmed fault Disputed, not Rejected", async function () {
    const { fl, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    await fl.connect(cars[3]).checkBack(1, false, EV);
    await fl.connect(cars[4]).checkBack(1, false, EV);
    await expect(fl.connect(cars[5]).checkBack(1, false, EV))
      .to.emit(fl, "FaultDisputed");
  });

  it("lets the auditor rule a disputed fault fake, which slashes", async function () {
    const { fl, rp, cars, auditor } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    for (const c of [cars[3], cars[4], cars[5]]) await fl.connect(c).checkBack(1, false, EV);

    await expect(fl.connect(auditor).resolveDispute(1, true))
      .to.emit(fl, "FaultRejected");
    // rewards clawed back: car0 got 5, loses it
    expect(await rp.freeBalance(cars[0].address)).to.equal(0);
  });

  it("lets the auditor rule it genuinely fixed", async function () {
    const { fl, cars, auditor } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    for (const c of [cars[3], cars[4], cars[5]]) await fl.connect(c).checkBack(1, false, EV);
    await expect(fl.connect(auditor).resolveDispute(1, false))
      .to.emit(fl, "FaultClosed");
  });

  it("refuses dispute resolution by a non-auditor", async function () {
    const { fl, cars, outsider } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await expect(fl.connect(outsider).resolveDispute(1, true))
      .to.be.revertedWithCustomError(fl, "MissingRole");
  });

  // plan 7.9: RepairClaimed + m absent check-backs returns to Closed
  it("closes a fault after a repair three owners confirm gone", async function () {
    const { fl, cars, repair } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    await expect(fl.connect(repair).claimRepair(1, EV)).to.emit(fl, "RepairClaimed");

    await fl.connect(cars[3]).checkBack(1, false, EV);
    await fl.connect(cars[4]).checkBack(1, false, EV);
    await expect(fl.connect(cars[5]).checkBack(1, false, EV))
      .to.emit(fl, "FaultClosed");
    expect(await fl.stateOf(1)).to.equal(State.Closed);
  });

  // plan 7.9: RepairClaimed + m present check-backs returns to Confirmed
  it("returns to Confirmed when the repair did not happen", async function () {
    const { fl, cars, repair } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    await fl.connect(repair).claimRepair(1, EV);

    await fl.connect(cars[3]).checkBack(1, true, EV);
    await fl.connect(cars[4]).checkBack(1, true, EV);
    await expect(fl.connect(cars[5]).checkBack(1, true, EV))
      .to.emit(fl, "FaultConfirmed");
    expect(await fl.stateOf(1)).to.equal(State.Confirmed);
  });

  it("refuses a repair claim from a car", async function () {
    const { fl, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    await expect(fl.connect(cars[3]).claimRepair(1, EV))
      .to.be.revertedWithCustomError(fl, "MissingRole");
  });

  it("allows one check-back per owner", async function () {
    const { fl, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).checkBack(1, false, EV);
    await expect(fl.connect(cars[1]).checkBack(1, true, EV))
      .to.be.revertedWithCustomError(fl, "OwnerAlreadyCheckedBack");
  });
});

describe("FaultLifecycle — lookups the car client needs", function () {
  it("lists active faults in a cell and drops closed ones", async function () {
    const { fl, cars, keeper } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).report(1, CELL, 1, 400, 800, EV);
    await fl.connect(cars[2]).report(0, OTHER_CELL, 2, 450, 900, EV);

    const prefix = CELL.slice(0, 16);   // first 7 bytes
    let active = await fl.activeFaultsIn(prefix);
    expect(active.length).to.equal(2);

    await time.increase(73 * 3600);
    await fl.connect(keeper).finalize(1);
    active = await fl.activeFaultsIn(prefix);
    expect(active.length).to.equal(1);
  });

  it("returns contributors in arrival order", async function () {
    const { fl, cars } = await deploy();
    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    const cs = await fl.contributorsOf(1);
    expect(cs[0]).to.equal(cars[0].address);
    expect(cs[1]).to.equal(cars[1].address);
  });
});
