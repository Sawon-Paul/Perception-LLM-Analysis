const { expect } = require("chai");
const { ethers } = require("hardhat");

// Batch 3: FLRoundLog and UnlearningLog.
// These carry the two security claims the thesis makes about the AI side:
// a car can detect a swapped model, and a removal can be checked by someone
// other than whoever performed it.

const h = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

async function deploy() {
  const s = await ethers.getSigners();
  const [admin, v2, v3, v4] = s;
  const cars = s.slice(4, 8);
  const server = s[10], unlearner = s[11], auditor = s[12], outsider = s[13];

  const Params = await ethers.getContractFactory("Params");
  const params = await Params.deploy([admin.address, v2.address, v3.address, v4.address]);
  const Registry = await ethers.getContractFactory("Registry");
  const registry = await Registry.deploy(admin.address, await params.getAddress());
  const RoadPoint = await ethers.getContractFactory("RoadPoint");
  const rp = await RoadPoint.deploy(admin.address, await registry.getAddress());
  await registry.setRoadPoint(await rp.getAddress());
  const Reputation = await ethers.getContractFactory("Reputation");
  const rep = await Reputation.deploy(admin.address, await params.getAddress());
  const FLRoundLog = await ethers.getContractFactory("FLRoundLog");
  const fl = await FLRoundLog.deploy(admin.address, await registry.getAddress(),
                                     await rep.getAddress());
  const UnlearningLog = await ethers.getContractFactory("UnlearningLog");
  const ul = await UnlearningLog.deploy(admin.address, await registry.getAddress());

  await registry.setRole(server.address, await registry.FL_SERVER(), true);
  await registry.setRole(unlearner.address, await registry.UNLEARN_SERVICE(), true);
  await registry.setRole(auditor.address, await registry.AUDITOR(), true);
  await rep.setUnlearningLog(await ul.getAddress());
  await ul.setRequester(await rep.getAddress(), true);
  for (let i = 0; i < cars.length; i++) {
    await registry.registerCar(cars[i].address, h(`o${i}`), h(`v${i}`));
  }
  return { admin, cars, server, unlearner, auditor, outsider,
           params, registry, rp, rep, fl, ul };
}

describe("FLRoundLog", function () {
  it("opens a round and records the base model", async function () {
    const { fl, server } = await deploy();
    await expect(fl.connect(server).openRound(h("base"), "round 1"))
      .to.emit(fl, "RoundOpened");
    expect(await fl.currentRound()).to.equal(1);
  });

  it("lets only the FL server open a round", async function () {
    const { fl, outsider } = await deploy();
    await expect(fl.connect(outsider).openRound(h("base"), ""))
      .to.be.revertedWithCustomError(fl, "NotServer");
  });

  it("refuses a second open round", async function () {
    const { fl, server } = await deploy();
    await fl.connect(server).openRound(h("base"), "");
    await expect(fl.connect(server).openRound(h("base2"), ""))
      .to.be.revertedWithCustomError(fl, "WrongState");
  });

  it("records a contribution with the weight read from Reputation", async function () {
    // The weight is read on-chain rather than supplied, so a server cannot
    // inflate a favoured client's influence after the fact.
    const { fl, server, cars } = await deploy();
    await fl.connect(server).openRound(h("base"), "");
    await expect(fl.connect(cars[0]).submitUpdate(1, h("upd0"), 500))
      .to.emit(fl, "UpdateSubmitted");
    const c = await fl.contributionOf(1, cars[0].address);
    expect(c.updateHash).to.equal(h("upd0"));
    expect(c.samples).to.equal(500);
    expect(c.weight).to.equal(500_000n);   // reputation 500/1000 scaled to 1e6
  });

  it("refuses two submissions from one car", async function () {
    const { fl, server, cars } = await deploy();
    await fl.connect(server).openRound(h("base"), "");
    await fl.connect(cars[0]).submitUpdate(1, h("a"), 10);
    await expect(fl.connect(cars[0]).submitUpdate(1, h("b"), 10))
      .to.be.revertedWithCustomError(fl, "AlreadySubmitted");
  });

  it("refuses a car excluded from FL after unlearning", async function () {
    const { fl, registry, admin, server, cars, unlearner } = await deploy();
    await registry.connect(admin).setRole(
      unlearner.address, await registry.UNLEARN_SERVICE(), true);
    await registry.connect(unlearner).setFLExclusion(cars[0].address, true);
    await fl.connect(server).openRound(h("base"), "");
    await expect(fl.connect(cars[0]).submitUpdate(1, h("x"), 10))
      .to.be.revertedWithCustomError(fl, "NotEligible");
  });

  it("refuses a deregistered car", async function () {
    const { fl, registry, admin, server, cars } = await deploy();
    await registry.connect(admin).deregisterCar(cars[0].address, "left");
    await fl.connect(server).openRound(h("base"), "");
    await expect(fl.connect(cars[0]).submitUpdate(1, h("x"), 10))
      .to.be.revertedWithCustomError(fl, "NotEligible");
  });

  it("records a rejection so a server cannot silently drop a client", async function () {
    const { fl, server, cars } = await deploy();
    await fl.connect(server).openRound(h("base"), "");
    await expect(fl.connect(server).rejectUpdate(1, cars[1].address, "outlier"))
      .to.emit(fl, "UpdateRejected").withArgs(1, cars[1].address, "outlier");
  });

  it("publishes the global hash on aggregation", async function () {
    const { fl, server, cars } = await deploy();
    await fl.connect(server).openRound(h("base"), "");
    await fl.connect(cars[0]).submitUpdate(1, h("u0"), 100);
    await fl.connect(cars[1]).submitUpdate(1, h("u1"), 200);
    await expect(fl.connect(server).aggregate(1, h("global1")))
      .to.emit(fl, "RoundAggregated").withArgs(1, h("global1"), 2, anyUint());
    expect(await fl.expectedModelHash(1)).to.equal(h("global1"));
  });

  // plan 13.1: model swap. The server serves a different file; the car checks
  // the hash against the chain and refuses.
  it("gives a car the hash it needs to detect a swapped model", async function () {
    const { fl, server, cars } = await deploy();
    await fl.connect(server).openRound(h("base"), "");
    await fl.connect(server).aggregate(1, h("honest_model"));

    const onChain = await fl.expectedModelHash(1);
    const downloaded = h("tampered_model");
    expect(downloaded).to.not.equal(onChain);
    await expect(fl.connect(cars[0]).reportMismatch(1, downloaded))
      .to.emit(fl, "ModelHashMismatch")
      .withArgs(1, cars[0].address, onChain, downloaded);
  });

  it("lists the rounds a car influenced, for unlearning", async function () {
    const { fl, server, cars } = await deploy();
    for (let r = 1; r <= 3; r++) {
      await fl.connect(server).openRound(h(`base${r}`), "");
      if (r !== 2) await fl.connect(cars[0]).submitUpdate(r, h(`u${r}`), 10);
      await fl.connect(server).aggregate(r, h(`g${r}`));
    }
    const rs = (await fl.roundsOf(cars[0].address)).map(Number);
    expect(rs).to.deep.equal([1, 3]);
    expect(await fl.firstRoundOf(cars[0].address)).to.equal(1);
  });

  it("refuses a zero hash", async function () {
    const { fl, server } = await deploy();
    await expect(fl.connect(server).openRound(ethers.ZeroHash, ""))
      .to.be.revertedWithCustomError(fl, "ZeroHash");
  });
});

describe("UnlearningLog", function () {
  it("records a request from a person", async function () {
    const { ul, admin, cars } = await deploy();
    await expect(ul.connect(admin).request(
      0, ethers.zeroPadValue(cars[0].address, 32), "owner asked"))
      .to.emit(ul, "UnlearnRequested");
    expect(await ul.requestCount()).to.equal(1);
  });

  // plan 7.9: two slashes trigger an automatic request
  it("accepts an automatic request from Reputation after two slashes", async function () {
    const { ul, rep, admin, cars } = await deploy();
    await rep.setController(admin.address, true);
    await rep.recordSlash(cars[0].address);
    await expect(rep.recordSlash(cars[0].address)).to.emit(ul, "UnlearnRequested");
    expect(await ul.requestCount()).to.equal(1);
  });

  it("refuses an automatic request from an unapproved contract", async function () {
    const { ul, outsider } = await deploy();
    await expect(ul.connect(outsider).requestFromContract(0, h("x"), "y"))
      .to.be.revertedWithCustomError(ul, "NotRequester");
  });

  it("issues a certificate an auditor can recheck", async function () {
    const { ul, admin, unlearner, cars } = await deploy();
    const subject = ethers.zeroPadValue(cars[0].address, 32);
    await ul.connect(admin).request(0, subject, "slashed twice");
    await ul.connect(unlearner).start(1, 2, 3);           // rollback from round 3
    await expect(ul.connect(unlearner).certify(
      1, 2, 3, 7, h("model_before"), h("model_after"), 120, 870, 40, 95))
      .to.emit(ul, "UnlearnCertified");

    const b = await ul.auditBundle(1);
    expect(b.before_).to.equal(h("model_before"));
    expect(b.after_).to.equal(h("model_after"));
    expect(b.fromRound).to.equal(3);
    expect(b.toRound).to.equal(7);
    expect(b.metricBefore).to.equal(870);   // attack success 87.0%
    expect(b.metricAfter).to.equal(40);     //               4.0%
  });

  it("lets an auditor pass or fail a certificate", async function () {
    const { ul, admin, unlearner, auditor, cars } = await deploy();
    await ul.connect(admin).request(0, ethers.zeroPadValue(cars[0].address, 32), "x");
    await ul.connect(unlearner).certify(
      1, 1, 1, 2, h("a"), h("b"), 10, 900, 50, 60);
    await expect(ul.connect(auditor).audit(1, true, "hashes recomputed, match"))
      .to.emit(ul, "CertificateAudited").withArgs(1, auditor.address, true,
                                                  "hashes recomputed, match");
    expect((await ul.certificates(1)).auditPassed).to.equal(true);
  });

  it("refuses an audit from anyone without the role", async function () {
    // The claim and its check must not come from the same party.
    const { ul, admin, unlearner, cars } = await deploy();
    await ul.connect(admin).request(0, ethers.zeroPadValue(cars[0].address, 32), "x");
    await ul.connect(unlearner).certify(1, 1, 1, 2, h("a"), h("b"), 1, 1, 1, 1);
    await expect(ul.connect(unlearner).audit(1, true, "self-approved"))
      .to.be.revertedWithCustomError(ul, "NotAuditor");
  });

  it("refuses certification by anyone but the unlearning service", async function () {
    const { ul, admin, outsider, cars } = await deploy();
    await ul.connect(admin).request(0, ethers.zeroPadValue(cars[0].address, 32), "x");
    await expect(ul.connect(outsider).certify(1, 1, 1, 2, h("a"), h("b"), 1, 1, 1, 1))
      .to.be.revertedWithCustomError(ul, "NotService");
  });

  it("refuses a zero model hash and a backwards round range", async function () {
    const { ul, admin, unlearner, cars } = await deploy();
    await ul.connect(admin).request(0, ethers.zeroPadValue(cars[0].address, 32), "x");
    await expect(ul.connect(unlearner).certify(
      1, 1, 1, 2, ethers.ZeroHash, h("b"), 1, 1, 1, 1))
      .to.be.revertedWithCustomError(ul, "ZeroHash");
    await expect(ul.connect(unlearner).certify(
      1, 1, 5, 2, h("a"), h("b"), 1, 1, 1, 1))
      .to.be.revertedWithCustomError(ul, "EmptyRange");
  });

  it("records a failure rather than leaving a request open", async function () {
    const { ul, admin, unlearner, cars } = await deploy();
    await ul.connect(admin).request(0, ethers.zeroPadValue(cars[0].address, 32), "x");
    await expect(ul.connect(unlearner).fail(1, "out of memory"))
      .to.emit(ul, "UnlearnFailed");
    expect((await ul.requests(1)).status).to.equal(3);
  });

  it("finds every request about one subject", async function () {
    const { ul, admin, cars } = await deploy();
    const s = ethers.zeroPadValue(cars[0].address, 32);
    await ul.connect(admin).request(0, s, "first");
    await ul.connect(admin).request(0, s, "second");
    expect((await ul.requestsFor(s)).map(Number)).to.deep.equal([1, 2]);
  });
});

function anyUint() {
  const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
  return anyValue;
}
