# Task: MEV Arbitrage
**Difficulty Level: ⭐⭐⭐⭐⭐ (5 stars)**

## Background

In this task, we explore **Maximal Extractable Value (MEV)** – the profit that can be extracted by strategically ordering blockchain transactions. The most common form is **DEX arbitrage**: exploiting price differences between decentralized exchanges to buy low on one and sell high on another.

MEV is a $600M+ annual market on Ethereum. Searchers (arbitrageurs) compete to find and execute profitable opportunities, often using flash loans to amplify capital without upfront funds. This task simulates a multi-DEX environment where price discrepancies exist, challenging you to design optimal arbitrage strategies.

The evaluation runs on Hardhat Network using mock DEX pools that implement real UniswapV2 constant-product AMM math. Your contract must identify and exploit arbitrage opportunities to maximize profit.

The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: Normalized profit ratio in range `[0.0, 1.0]`.
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Comprehensive execution feedback including actual profit/loss amounts, pool prices before/after trades, price impacts (slippage), and actionable hints. This helps the LLM learn from specific failures—especially understanding that large trade sizes cause excessive slippage that can turn potential profits into losses.

---

## Task Description

**Goal:** Design a Solidity contract named `Arbitrage` that extracts maximum profit from price discrepancies across multiple DEX pools.

**Setup:**
- 4 DEX pools with different token pairs and deep liquidity
- Large price discrepancies between pools (~17.6% spread)
- Flash loan provider (10,000 WETH available, 0.05% fee)
- Starting capital: 10 WETH

**Pool Configuration:**

| Pool | Pair | Reserves | Implied Price |
|------|------|----------|---------------|
| Pool A | WETH/USDC | 5,000 WETH / 8,500,000 USDC | 1 WETH = **1,700** USDC (CHEAP) |
| Pool B | WETH/USDC | 4,000 WETH / 8,000,000 USDC | 1 WETH = **2,000** USDC (EXPENSIVE) |
| Pool C | WETH/DAI | 3,000 WETH / 5,550,000 DAI | 1 WETH = 1,850 DAI |
| Pool D | USDC/DAI | 5,000,000 USDC / 5,100,000 DAI | 1 USDC = 1.02 DAI |


**Requirements:**
1. Contract must be named `Arbitrage`
2. Must have `executeArbitrage()` function (main entry point)
3. Must have `onFlashLoan()` callback for flash loans (optional but recommended)
4. Must have `getProfit()` view function returning profit in WETH
5. Must have `receive()` function for ETH
6. Constructor accepts: flashLender, pools array, tokens array
7. Must compile with Solidity ^0.8.0
8. **Flash loan repayment**: `onFlashLoan()` must use `transfer()` (NOT `approve()`) to repay:
   ```solidity
   // CORRECT - transfer tokens back to lender
   IERC20(weth).transfer(flashLender, amount + fee);
   
   // WRONG - approve() doesn't transfer tokens, loan will fail!
   IERC20(weth).approve(flashLender, amount + fee);
   ```

**Things to Avoid:**
1. Running out of gas during complex multi-hop swaps
2. Incorrect slippage calculations leading to losses
3. Failing to repay flash loans (will revert)
4. Hardcoding values that only work for specific pool states

**Key Insights:**
- Pools use UniswapV2 constant-product formula (x * y = k) with 0.3% swap fee
- **Critical**: Use batched trades to minimize slippage - single large trades lose money!
- Flash loans available at 0.05% fee (5 basis points)
- Deep liquidity pools (5,000+ WETH) allow large-scale extraction
- Multiple arbitrage paths may exist - find the optimal combination

---

## AMM Math Reference

Pools use **UniswapV2 constant-product formula**: `x * y = k`

```solidity
// Calculate output amount for a swap
function getAmountOut(uint amountIn, uint reserveIn, uint reserveOut) 
    public pure returns (uint amountOut) 
{
    uint amountInWithFee = amountIn * 997;  // 0.3% swap fee
    uint numerator = amountInWithFee * reserveOut;
    uint denominator = (reserveIn * 1000) + amountInWithFee;
    amountOut = numerator / denominator;
}
```

**Example calculation:**
- Swap 1 WETH in Pool A (5,000 WETH, 8,500,000 USDC)
- `amountOut = (1 * 997 * 8500000) / (5000 * 1000 + 1 * 997)`
- `amountOut = 8,474,500,000 / 5,000,997 ≈ 1,694 USDC`
- Effective price: ~1,694 USDC/WETH (minimal slippage with small trade)

**Slippage Warning:** Large trades cause significant price impact. A 100 WETH trade would move the price by ~2%, eating into profits.

---

## Reward Description

```python
def compute_reward(final_weth: float, initial_weth: float = 10.0) -> float:
    profit = final_weth - initial_weth
    max_extractable = 20.0  # Maximum profit with optimal flash loan strategy
    reward = profit / max_extractable
    return max(0.0, min(1.0, reward))
```

---

## Initial Code

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
}

interface IDEXPool {
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1);
    function swap(uint amount0Out, uint amount1Out, address to) external;
}

/**
 * @title Arbitrage
 * @notice BASELINE - Simple single-trade arbitrage (lots of room for improvement!)
 * 
 * Environment (deep liquidity pools for progressive optimization):
 *   - Pool A: WETH/USDC at 1700 USDC/WETH (5,000 WETH) - CHEAP, BUY HERE
 *   - Pool B: WETH/USDC at 2000 USDC/WETH (4,000 WETH) - EXPENSIVE, SELL HERE
 *   - Pool C: WETH/DAI at 1850 DAI/WETH (triangular arbitrage)
 *   - Flash lender: 10,000 WETH available at 0.05% fee
 * 
 * Starting capital: 10 WETH | Spread: 17.6%
 * 
 * KNOWN ISSUES (intentionally suboptimal):
 *   - Only trades 2 WETH (leaves 80% unused!)
 *   - Single trade (no iteration)
 *   - No flash loans (could amplify 1000x!)
 * 
 * IMPROVEMENT PATH:
 *   1. Batched trades → Tier 2 (~0.6 reward)
 *   2. Flash loans → Tier 4 (~0.7+ reward)
 */
contract Arbitrage {
    address public owner;
    address public flashLender;
    address[] public pools;
    address public weth;
    address public usdc;
    address public dai;
    
    constructor(
        address _flashLender,
        address[] memory _pools,
        address[] memory _tokens
    ) {
        owner = msg.sender;
        flashLender = _flashLender;
        pools = _pools;
        weth = _tokens[0];
        usdc = _tokens[1];
        dai = _tokens[2];
    }
    
    function executeArbitrage() external {
        IDEXPool poolA = IDEXPool(pools[0]);  // Cheap WETH (1700)
        IDEXPool poolB = IDEXPool(pools[1]);  // Expensive WETH (2000)
        
        // SUBOPTIMAL: Only trade 2 WETH once (leaving 8 WETH unused!)
        uint256 tradeAmount = 2e18;  // 2 WETH
        
        // Sell WETH in Pool B (where it's expensive)
        (uint112 r0B, uint112 r1B) = poolB.getReserves();
        uint256 usdcOut = _getAmountOut(tradeAmount, r0B, r1B);
        IERC20(weth).transfer(address(poolB), tradeAmount);
        poolB.swap(0, usdcOut, address(this));
        
        // Buy WETH in Pool A (where it's cheap)
        (uint112 r0A, uint112 r1A) = poolA.getReserves();
        uint256 usdcBal = IERC20(usdc).balanceOf(address(this));
        uint256 wethOut = _getAmountOut(usdcBal, r1A, r0A);
        IERC20(usdc).transfer(address(poolA), usdcBal);
        poolA.swap(wethOut, 0, address(this));
    }
    
    function _getAmountOut(uint amountIn, uint reserveIn, uint reserveOut) 
        internal pure returns (uint) 
    {
        if (amountIn == 0 || reserveIn == 0 || reserveOut == 0) return 0;
        uint amountInWithFee = amountIn * 997;
        uint numerator = amountInWithFee * reserveOut;
        uint denominator = (reserveIn * 1000) + amountInWithFee;
        return numerator / denominator;
    }
    
    function getProfit() external view returns (uint256) {
        return IERC20(weth).balanceOf(address(this));
    }
    
    // Flash loan callback - implement your strategy here!
    function onFlashLoan(
        address, address, uint256, uint256, bytes calldata
    ) external pure returns (bytes32) {
        return keccak256("ERC3156FlashBorrower.onFlashLoan");
    }
    
    receive() external payable {}
}
```

---

## File Structure

```
iterx_code/mev_arbitrage/
├── README.md                    # This file
├── __init__.py                  # Package exports
├── initial_code.sol             # Baseline arbitrage contract
├── eval_mev_arbitrage.py        # Evaluation function
├── run_iterx.py                 # IterX evaluation runner
├── contracts/
│   ├── MockERC20.sol            # ERC20 token implementation
│   ├── MockDEXPool.sol          # UniswapV2-style AMM pool
│   └── MockFlashLender.sol      # EIP-3156 flash lender
└── hardhat/
    ├── hardhat.config.js        # Hardhat configuration
    └── arbitrage_test.js        # Test harness
```

---

## Running iterX

```bash
python run_iterx.py
```
