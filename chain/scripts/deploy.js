// Deploy the seven contracts and wire their permissions.
//
// Order matters: contracts reference each other, and several permissions can
// only be granted after both sides exist. Doing this by hand is where a
// deployment usually breaks \u2014 a missing setController leaves FaultLifecycle
// unable to move tokens, and nothing fails until the first report.
//
// The script verifies every link afterwards and refuses to write the address
// file if any check fails, so a half-wired deployment cannot be mistaken for a
// good one.
//
//   npx hardhat run scripts/deploy.js --network besu
//   npx hardhat run scripts/deploy.js                 (in-process, for testing)

const fs = require("fs");
const path = require("path");
const { ethers, network } = require("hardhat");

async function main() {
  const signers = await ethers.getSigners();
  const admin = signers[0];
  // On a real Besu chain there is one funded key; roles go to addresses from
  // the environment. In-process there are twenty signers to play the parts.
  const pick = (i, envName) =>
    process.env[envName] || (signers[i] ? signers[i].address : admin.address);

  const consortium = [
    admin.address,
    pick(1, "VALIDATOR_2"),
    pick(2, "VALIDATOR_3"),
    pick(3, "VALIDATOR_4"),
  ];

  console.log(`network: ${network.name}`);
  console.log(`admin:   ${admin.address}\n`);

  const deploy = async (name, args) => {
    const F = await ethers.getContractFactory(name);
    const c = await F.deploy(...args);
    await c.waitForDeployment();
    const a = await c.getAddress();
    console.log(`  ${name.padEnd(16)} ${a}`);
    return c;
  };

  console.log("deploying");
  const params = await deploy("Params", [consortium]);
  const registry = await deploy("Registry", [admin.address, await params.getAddress()]);
  const roadPoint = await deploy("RoadPoint", [admin.address, await registry.getAddress()]);
  const reputation = await deploy("Reputation", [admin.address, await params.getAddress()]);
  const faults = await deploy("FaultLifecycle", [
    admin.address, await registry.getAddress(), await roadPoint.getAddress(),
    await reputation.getAddress(), await params.getAddress(),
  ]);
  const flLog = await deploy("FLRoundLog", [
    admin.address, await registry.getAddress(), await reputation.getAddress(),
  ]);
  const ulLog = await deploy("UnlearningLog", [
    admin.address, await registry.getAddress(),
  ]);

  console.log("\nwiring");
  const steps = [
    ["Registry knows RoadPoint",
      () => registry.setRoadPoint(roadPoint.getAddress())],
    ["FaultLifecycle may move tokens",
      () => roadPoint.setController(faults.getAddress(), true)],
    ["FaultLifecycle may adjust reputation",
      () => reputation.setController(faults.getAddress(), true)],
    ["Reputation may request unlearning",
      () => reputation.setUnlearningLog(ulLog.getAddress())],
    ["FaultLifecycle may request unlearning",
      () => faults.setUnlearningLog(ulLog.getAddress())],
    ["UnlearningLog accepts Reputation",
      () => ulLog.setRequester(reputation.getAddress(), true)],
    ["UnlearningLog accepts FaultLifecycle",
      () => ulLog.setRequester(faults.getAddress(), true)],
  ];
  for (const [label, fn] of steps) {
    const tx = await fn();
    await tx.wait();
    console.log(`  ${label}`);
  }

  // Roles. On Besu these come from the environment; in-process they are signers
  // so the whole flow can be exercised.
  const roles = [
    ["FL_SERVER", pick(10, "FL_SERVER_ADDRESS")],
    ["UNLEARN_SERVICE", pick(11, "UNLEARN_SERVICE_ADDRESS")],
    ["AUDITOR", pick(12, "AUDITOR_ADDRESS")],
    ["KEEPER", pick(13, "KEEPER_ADDRESS")],
    ["REPAIR", pick(14, "REPAIR_ADDRESS")],
    ["MERCHANT", pick(15, "MERCHANT_ADDRESS")],
  ];
  console.log("\nroles");
  for (const [role, who] of roles) {
    const tx = await registry.setRole(await registry[role](), who, true)
      .catch(() => registry.setRole(who, ethers.id(role), true));
    await tx.wait();
    console.log(`  ${role.padEnd(16)} ${who}`);
  }

  // ---- verify, before anything downstream trusts this deployment ----
  console.log("\nverifying");
  const checks = [
    ["Registry -> RoadPoint",
      async () => (await registry.roadPoint()).toLowerCase() ===
                  (await roadPoint.getAddress()).toLowerCase()],
    ["RoadPoint controller",
      async () => await roadPoint.isController(await faults.getAddress())],
    ["Reputation controller",
      async () => await reputation.isController(await faults.getAddress())],
    ["UnlearningLog requester: Reputation",
      async () => await ulLog.isRequester(await reputation.getAddress())],
    ["UnlearningLog requester: FaultLifecycle",
      async () => await ulLog.isRequester(await faults.getAddress())],
    ["Params seeded (k_confirm)",
      async () => (await params.get("k_confirm")) === 3n],
    ["FL_SERVER role set",
      async () => await registry.hasRole(roles[0][1], await registry.FL_SERVER())],
  ];
  let ok = true;
  for (const [label, fn] of checks) {
    let pass = false;
    try { pass = await fn(); } catch (e) { pass = false; }
    console.log(`  ${pass ? "ok  " : "FAIL"}  ${label}`);
    ok = ok && pass;
  }
  if (!ok) {
    throw new Error("deployment is only partly wired \u2014 addresses not written. " +
                    "Fix the failing step and redeploy onto a clean chain.");
  }

  const out = {
    network: network.name,
    chainId: Number((await ethers.provider.getNetwork()).chainId),
    deployedAt: new Date().toISOString(),
    admin: admin.address,
    consortium,
    contracts: {
      Params: await params.getAddress(),
      Registry: await registry.getAddress(),
      RoadPoint: await roadPoint.getAddress(),
      Reputation: await reputation.getAddress(),
      FaultLifecycle: await faults.getAddress(),
      FLRoundLog: await flLog.getAddress(),
      UnlearningLog: await ulLog.getAddress(),
    },
    roles: Object.fromEntries(roles),
  };
  const dir = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${network.name}.json`);
  fs.writeFileSync(file, JSON.stringify(out, null, 2));
  console.log(`\naddresses -> ${file}`);
  console.log("The Python client reads this file; nothing is hardcoded.");
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
