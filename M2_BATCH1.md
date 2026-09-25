# M2 batch 1 — Params, Registry, RoadPoint

Compiled and tested. **29 tests passing.**

## Install

```cmd
cd /d C:\Sawon\LLM
```

Copy `chain\` into the project, then:

```cmd
cd chain
npm install
npx hardhat test
```

Expect 29 passing.

## Two build decisions worth knowing

**Solidity 0.8.24, EVM target Shanghai.** OpenZeppelin 5.1 and later emit the
`mcopy` opcode, which exists only after the Cancun fork. If the Besu genesis
does not enable Cancun, contracts built against it deploy and then revert at
runtime — a hard bug to trace. OpenZeppelin is pinned to 5.0.2 and the EVM
target to Shanghai, so the bytecode is valid on a plain QBFT chain.

**The compiler comes from npm.** `hardhat.config.js` overrides the solc-download
subtask to use `node_modules/solc/soljson.js`. If your machine can reach
`binaries.soliditylang.org` you can delete that override; it does no harm either
way.

## Params — section 7.2

All tunable numbers live here, so a sensitivity test is a transaction rather
than a redeployment.

Changing one needs 3 of 4 consortium approvals. A member cannot approve twice,
so no single organisation can lower the stake or lower `k` on its own. That is
the on-chain governance the thesis argues for.

`get()` reverts on an unknown name rather than returning 0. A silent zero for
`stake` or `k_confirm` would switch off a safety rule with nothing in the logs.

Starting values are those in plan section 15, plus `registration_bond = 25`
(five parallel reports at a stake of 5) and `rep_threshold_unlearn = 200`.

## Registry — section 7.3

One registration per vehicle registration hash. This is the Sybil defence:
confirmation needs k *distinct owners*, so if identities were free an attacker
could confirm any fake fault alone. The vehicle document is the scarce thing.

Owner identity is stored as a hash. The chain is publicly readable, so a name
or a document number on it would publish who drives where.

The vehicle hash stays claimed after deregistration. Releasing it would let an
owner leave and rejoin to shed a bad reputation, which would undo the entire
penalty system.

`sameOwner(a, b)` is what `FaultLifecycle` will call for the distinct-owner
rule, and it returns false for unregistered addresses rather than treating two
empty owner ids as a match.

## RoadPoint — section 7.4

Not a cryptocurrency, and closed off deliberately: `transfer`, `transferFrom`
and `approve` all revert. The only exit is redemption at a registered merchant,
which burns the points. No market, no on-chain rate, no car-to-car value
movement.

Balances are split. **Bond** is minted at registration, backs stakes, and can
never be redeemed. **Earned** points are what a car can spend. `freeBalance()`
excludes both bond and locked stake.

Redemption is signed by the car and submitted by the merchant, so the merchant
cannot drain a balance alone. A nonce blocks the double-redeem attack from
section 13.1: a replayed signature carries a stale nonce and reverts.

Clawback burns what it can and records the rest as debt, which future rewards
repay before the car sees anything. It cannot touch the bond.

## Tests, mapped to the plan

| plan | test |
|---|---|
| 7.9 car-to-car transfer reverts | ✔ plus approve and transferFrom |
| 7.9 redeem by a non-merchant reverts | ✔ |
| 7.9 same owner cannot count twice toward k | ✔ `sameOwner` |
| 7.9 setParam fails with 2 approvals | ✔ and succeeds on 3 |
| 13.1 Sybil: same vehicle twice | ✔ reverts |
| 13.1 double redeem | ✔ replayed signature reverts |

Extra cases not in the plan but needed: re-registering after leaving, clawback
against the bond, locking more than held, unknown parameter names, and
withdrawing an approval.

## Not done in this batch

`FaultLifecycle` is not written yet, so a plain signer stands in for it as the
token controller. When batch 2 lands, point `setController` at the real
contract address instead.

## Next

Batch 2: `Reputation` and `FaultLifecycle` — the state machine, the
distinct-owner rule, rewards, slashing and expiry. It is the largest piece.
