# Task: Smart Contract Gas Optimization in Blockchain
**Difficulty Level: ⭐⭐⭐ (3 stars)**

---

## Background

In this task, we will design a gas-efficient smart contract for batch ERC-20 token transfers.
Batch operations are fundamental to blockchain applications, as airdrops, payroll systems, reward distributions, and DAO treasury management all require sending tokens to multiple addresses efficiently. On Ethereum, every operation costs gas, and gas costs real money. A single airdrop to 10,000 addresses using a naive implementation could cost thousands of dollars more than an optimized version.

The evaluation runs on Hardhat Network, a local EVM implementation that accurately measures gas consumption. Your contract is deployed and executed on this local EVM. This is free to run (no real ETH required) while providing accurate gas measurements representative of mainnet behavior. The setup can be easily extended to deploy on real networks (testnets or mainnet) by simply changing the `--network` flag. The optimization principles focus on minimizing storage operations, using efficient loops, and leveraging assembly where appropriate.

The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: Inverse of total gas used (`1e9 / total_gas`).
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Always `""`.

---

## Task Description

**Goal:** Design a gas-minimizing batch transfer function for ERC-20 tokens.

**Input:** 
- `token`: Address of the ERC-20 token contract
- `recipients`: Array of recipient addresses
- `amounts`: Array of token amounts corresponding to each recipient

**Output:**
- All transfers executed successfully
- Minimal gas consumed

**Requirements:**
1. Function must be named `batchTransfer`
2. Must handle arbitrary batch sizes (1 to 1000+ recipients)
3. Must revert if any individual transfer fails
4. Must emit a `BatchTransferCompleted` event with total amount and recipient count
5. Must be compatible with standard ERC-20 tokens
6. No external dependencies beyond OpenZeppelin (optional)

**Things to Avoid:**
1. Unsafe external calls without proper checks
2. Integer overflow/underflow (use Solidity 0.8+)
3. Reentrancy vulnerabilities
4. Exceeding block gas limits for large batches

## Reward Description

The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: Inverse of total gas used (`1e9 / total_gas`).
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Always `""`.

The reward is computed as:

```
reward = 1e9 / total_gas
```

Where `total_gas` is the sum of gas used across all batch sizes [10, 50, 100, 500].

- **Higher reward = better** (less gas = higher reward)
- Returns `0` if correctness tests fail

## Initial Code

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract BatchTransfer {
    event BatchTransferCompleted(
        address indexed token,
        address indexed sender,
        uint256 totalAmount,
        uint256 recipientCount
    );
    
    /**
     * @notice Transfers tokens to multiple recipients
     * @param token The ERC20 token address
     * @param recipients Array of recipient addresses
     * @param amounts Array of amounts to transfer
     */
    function batchTransfer(
        address token,
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external {
        require(recipients.length == amounts.length, "Length mismatch");
        require(recipients.length > 0, "Empty arrays");
        
        IERC20 tokenContract = IERC20(token);
        uint256 totalAmount = 0;
        
        for (uint256 i = 0; i < recipients.length; i++) {
            require(recipients[i] != address(0), "Zero address");
            require(amounts[i] > 0, "Zero amount");
            
            totalAmount += amounts[i];
            
            bool success = tokenContract.transferFrom(
                msg.sender,
                recipients[i],
                amounts[i]
            );
            require(success, "Transfer failed");
        }
        
        emit BatchTransferCompleted(token, msg.sender, totalAmount, recipients.length);
    }
}
```

## File Structure

```
smart_contract/
├── README.md                    # This file
├── eval_smart_contract_gas.py   # Main evaluation script
├── initial_code.sol             # Initial implementation to optimize
├── run_iterx.py                 # IterX evaluation runner
├── contracts/
│   ├── MockERC20.sol            # Test token for evaluation
│   └── interfaces/
│       └── IERC20.sol           # ERC20 interface
├── correctness_check/
│   └── BatchTransfer.test.js    # Correctness tests
└── hardhat/
    ├── config.json              # Evaluation parameters
    ├── gas_benchmark.js         # Hardhat gas measurement
    └── hardhat.config.js        # Hardhat configuration
```

## Running Evaluation

```python
from eval_smart_contract_gas import get_reward

# Get reward for a submission
reward = get_reward('path/to/your/BatchTransfer.sol', 'path/to/work/dir')
print(reward)  # e.g., 51.53 (higher = better)
```

---

## Running iterX

```bash
python run_iterx.py
```

