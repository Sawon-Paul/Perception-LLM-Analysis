# M3 — the consortium chain

Four Besu validators running QBFT, free gas, and the seven contracts deployed
onto them.

## Why these choices

**QBFT, not proof of work.** A permissioned chain has known validators, so
there is nothing to mine. Blocks finalise in one round and there are no forks
to wait out.

**Free gas (`zeroBaseFee`, `--min-gas-price=0`).** Cars must transact without
holding a native coin. Cryptocurrency trading is illegal in Bangladesh, so any
design where a vehicle needs ETH to file a report is unusable there.

**Shanghai EVM.** Matches the `evmVersion` the contracts compile to. A mismatch
deploys fine and then reverts at runtime, which is painful to trace.

**Only node 1 exposes RPC.** The others are reachable inside the Docker
network only. That is what a consortium looks like, and it keeps the fault
tolerance test honest.

## Run it

Docker must be running.

**1. Generate the genesis and validator keys.** Besu does this itself — the
QBFT `extraData` field is RLP-encoded validator data, and hand-writing it
produces a chain that starts and then never makes a block.

```cmd
cd C:\Sawon\LLM\chain
docker run --rm -v "%cd%/besu:/cfg" hyperledger/besu:latest operator generate-blockchain-config --config-file=/cfg/qbft-config.json --to=/cfg/networkFiles --private-key-file-name=key
```

**2. Arrange the node directories.**

```cmd
node besu/prepare.js
```

Writes `besu/nodes/node1..4`, each with its key and a `static-nodes.json`
naming the other three. Without that file the validators start, find no peers
and sit at block 0 looking healthy. It also writes `besu/.env` with the
validator addresses and the deployer key.

`besu/.env` holds a private key. Add it to `.gitignore`.

**3. Start the chain.**

```cmd
docker compose -f besu/docker-compose.yml up -d
docker compose -f besu/docker-compose.yml ps
```

**4. Check it.**

```cmd
node besu/verify.js
```

Expect 4 validators and about 0.5 blocks per second.

**5. Deploy.**

```cmd
npx hardhat run scripts/deploy.js --network besu
```

Deploys all seven contracts, wires the permissions, verifies every link, and
writes `chain/deployments/besu.json`. If any check fails it writes nothing —
a half-wired deployment cannot be mistaken for a good one.

## The fault tolerance experiment

QBFT needs more than two thirds of validators. With 4, that means 3 must stay
up, so exactly **one** failure is tolerated. Measure it rather than asserting
it:

```cmd
docker stop besu-node4
node besu/verify.js
docker stop besu-node3
node besu/verify.js
docker start besu-node3 besu-node4
node besu/verify.js
```

Expect: normal, normal, halted, recovered. Record the block rate at each step —
that table is the availability result for the thesis.

## Contracts

| contract | role |
|---|---|
| Params | every tunable number, changed only by 3-of-4 consortium approval |
| Registry | car identity, roles, one registration per vehicle document |
| RoadPoint | non-transferable points; stake, slash, redeem at a merchant |
| Reputation | one score driving both rewards and FL weight |
| FaultLifecycle | report, confirm, check-back, repair, expiry, slashing |
| FLRoundLog | model hashes per round, so a swapped model is detectable |
| UnlearningLog | removal requests and certificates a third party can recheck |

84 tests pass, including the Sybil, collusion, double-redeem and model-swap
cases from plan section 13.1.
