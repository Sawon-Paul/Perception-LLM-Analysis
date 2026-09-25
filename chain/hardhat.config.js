require("@nomicfoundation/hardhat-toolbox");
const path = require("path");
const { subtask } = require("hardhat/config");
const {
  TASK_COMPILE_SOLIDITY_GET_SOLC_BUILD,
} = require("hardhat/builtin-tasks/task-names");

// besu/.env holds the deployer key, written by besu/prepare.js. It only exists
// after the chain has been generated, and dotenv is only needed to read it — so
// a missing package or missing file must not break `hardhat test`, which needs
// neither.
try {
  require("dotenv").config({ path: path.join(__dirname, "besu", ".env") });
} catch (e) {
  if (e.code !== "MODULE_NOT_FOUND") throw e;
  // no dotenv installed: fine until you deploy to Besu, which needs it
}

// The solc binary host is not always reachable, so use the compiler that ships
// with the npm `solc` package instead of downloading one.
subtask(TASK_COMPILE_SOLIDITY_GET_SOLC_BUILD, async (args, hre, runSuper) => {
  if (args.solcVersion === "0.8.24") {
    return {
      compilerPath: path.join(__dirname, "node_modules", "solc", "soljson.js"),
      isSolcJs: true,
      version: args.solcVersion,
      longVersion: "0.8.24",
    };
  }
  return runSuper();
});

module.exports = {
  solidity: {
    version: "0.8.24",
    settings: {
      optimizer: { enabled: true, runs: 200 },
      // Shanghai, not Cancun. OpenZeppelin 5.1+ emits the `mcopy` opcode, which
      // only exists after Cancun; if the Besu genesis does not enable that fork
      // the contracts deploy and then revert at runtime. Pinning both the EVM
      // version and OpenZeppelin 5.0.2 keeps the bytecode valid on a plain
      // QBFT chain.
      evmVersion: "shanghai",
    },
  },
  networks: {
    // Besu QBFT, free gas: cars hold no native coin.
    // The deployer key comes from besu/.env so no key is ever committed.
    besu: {
      url: process.env.BESU_RPC || "http://127.0.0.1:8545",
      gasPrice: 0,
      chainId: Number(process.env.BESU_CHAIN_ID || 1337),
      accounts: process.env.DEPLOYER_KEY ? [process.env.DEPLOYER_KEY] : [],
    },
  },
};
