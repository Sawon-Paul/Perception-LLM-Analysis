const { expect } = require("chai");
const { ethers } = require("hardhat");
const h = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));
const CELL = "0x3667356466346e75", EV = h("e");

describe("fraud economics", function () {
  it("measures what a colluding group of 3 ends up with", async function () {
    const s = await ethers.getSigners();
    const [admin, v2, v3, v4] = s;
    const cars = s.slice(4, 12), auditor = s[13];
    const P = await ethers.getContractFactory("Params");
    const params = await P.deploy([admin.address,v2.address,v3.address,v4.address]);
    const R = await ethers.getContractFactory("Registry");
    const registry = await R.deploy(admin.address, await params.getAddress());
    const RP = await ethers.getContractFactory("RoadPoint");
    const rp = await RP.deploy(admin.address, await registry.getAddress());
    await registry.setRoadPoint(await rp.getAddress());
    const RE = await ethers.getContractFactory("Reputation");
    const rep = await RE.deploy(admin.address, await params.getAddress());
    const F = await ethers.getContractFactory("FaultLifecycle");
    const fl = await F.deploy(admin.address, await registry.getAddress(),
      await rp.getAddress(), await rep.getAddress(), await params.getAddress());
    await rp.setController(await fl.getAddress(), true);
    await rep.setController(await fl.getAddress(), true);
    await registry.setRole(auditor.address, await registry.AUDITOR(), true);
    for (let i=0;i<cars.length;i++)
      await registry.registerCar(cars[i].address, h(`o${i}`), h(`v${i}`));

    const bal = async (c) => await rp.balanceOf(c.address);
    const attackers = [cars[0], cars[1], cars[2]];
    let before = 0n; for (const c of attackers) before += await bal(c);

    await fl.connect(cars[0]).report(0, CELL, 2, 450, 900, EV);
    await fl.connect(cars[1]).confirm(1, 2, EV);
    await fl.connect(cars[2]).confirm(1, 2, EV);
    let afterReward = 0n; for (const c of attackers) afterReward += await bal(c);

    for (const c of [cars[3],cars[4],cars[5]]) await fl.connect(c).checkBack(1,false,EV);
    await fl.connect(auditor).resolveDispute(1, true);

    let after = 0n; for (const c of attackers) after += await bal(c);
    let reps = []; for (const c of attackers) reps.push(await rep.get(c.address));
    console.log(`\n      attacker tokens before the attack : ${before}`);
    console.log(`      after the fake fault confirmed   : ${afterReward}  (+${afterReward-before})`);
    console.log(`      after it was ruled fake          : ${after}  (${after-before} vs start)`);
    console.log(`      reputations now                  : ${reps.join(", ")} (started 500)`);
    expect(after).to.be.lessThan(before);
  });
});
