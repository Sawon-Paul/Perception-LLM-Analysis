// Check the chain is alive and Byzantine fault tolerant.
//
// Plan M3 done-test: blocks every ~2 s; stopping one validator keeps them
// coming; stopping two halts cleanly. QBFT needs more than two thirds of
// validators, so 4 nodes tolerate exactly 1 failure — that is measured here
// rather than assumed.
//
//   node besu/verify.js
//   node besu/verify.js --seconds 30
const RPC = process.env.BESU_RPC || "http://127.0.0.1:8545";

// Accept --seconds=30 and --seconds 30, and fall back to 20 on anything that
// is not a positive number. The earlier version produced NaN, waited 1 ms, saw
// no new block and reported a halt on a perfectly healthy chain.
function arg(name, dflt) {
  const a = process.argv.slice(2);
  let raw = null;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === `--${name}`) raw = a[i + 1];
    else if (a[i].startsWith(`--${name}=`)) raw = a[i].split("=")[1];
  }
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : dflt;
}
const seconds = arg("seconds", 20);

async function rpc(method, params = []) {
  const r = await fetch(RPC, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  });
  const j = await r.json();
  if (j.error) throw new Error(`${method}: ${j.error.message}`);
  return j.result;
}

(async () => {
  console.log(`rpc: ${RPC}\n`);
  let validators = [];
  try {
    console.log(`client      ${await rpc("web3_clientVersion")}`);
    console.log(`chain id    ${parseInt(await rpc("eth_chainId"), 16)}`);
    validators = await rpc("qbft_getValidatorsByBlockNumber", ["latest"]);
    console.log(`validators  ${validators.length}`);
    validators.forEach(v => console.log(`            ${v}`));
    const needed = Math.floor((2 * validators.length) / 3) + 1;
    console.log(`\nQBFT needs more than 2/3 of ${validators.length} validators: ` +
                `${needed} must stay up, so ${validators.length - needed} ` +
                `failure(s) are tolerated.`);
  } catch (e) {
    console.error(`cannot reach the chain: ${e.message}`);
    console.error("Is it running?  docker compose -f besu/docker-compose.yml ps");
    process.exit(1);
  }

  // how many peers node 1 can see, which is what actually fails when a
  // validator is stopped
  try {
    const peers = parseInt(await rpc("net_peerCount"), 16);
    console.log(`peers seen by node 1: ${peers} of ${validators.length - 1}`);
  } catch { /* ADMIN/NET api may be off; not fatal */ }

  const start = parseInt(await rpc("eth_blockNumber"), 16);
  console.log(`\nwatching for ${seconds}s from block ${start} ...`);
  await new Promise(r => setTimeout(r, seconds * 1000));
  const end = parseInt(await rpc("eth_blockNumber"), 16);
  const made = end - start;
  const rate = made / seconds;
  console.log(`block ${end}: ${made} blocks in ${seconds}s = ${rate.toFixed(2)}/s` +
              (made ? ` (${(1 / rate).toFixed(1)}s per block)` : ""));

  if (made === 0) {
    console.log("\nHALTED — no block in the whole window.");
    console.log("Expected once two of four validators are down. If all four are");
    console.log("up, check peering in besu/nodes/*/data/static-nodes.json.");
  } else if (rate < 0.3) {
    console.log("\nSlow. Some validators are probably unreachable.");
  } else {
    console.log("\nProducing normally.");
  }
  console.log("\nFault tolerance test, for the thesis:");
  console.log("  docker stop besu-node4              # 3 of 4 — keeps going");
  console.log("  node besu/verify.js");
  console.log("  docker stop besu-node3              # 2 of 4 — halts");
  console.log("  node besu/verify.js");
  console.log("  docker start besu-node3 besu-node4  # recovers");
})();
