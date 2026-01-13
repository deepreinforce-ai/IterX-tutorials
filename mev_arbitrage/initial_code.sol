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
 *   - Pool A: WETH/USDC at 1700 USDC/WETH (5,000 WETH liquidity) - CHEAP, BUY HERE
 *   - Pool B: WETH/USDC at 2000 USDC/WETH (4,000 WETH liquidity) - EXPENSIVE, SELL HERE
 *   - Pool C: WETH/DAI at 1850 DAI/WETH (3,000 WETH - for triangular arbitrage)
 *   - Pool D: USDC/DAI (stablecoin pool)
 *   - Flash lender: 10,000 WETH available at 0.05% fee
 * 
 * Starting capital: 10 WETH
 * Spread: 17.6% between Pool A and Pool B
 * 
 * 
 * IMPROVEMENT PATH:
 *   1. Use all 10 WETH in multiple smaller batches → Tier 2 (~0.6 reward)
 *   2. Optimize batch sizes based on spread → Tier 3 (~0.65 reward)
 *   3. Add flash loans (borrow 100-500 WETH) → Tier 4 (~0.7+ reward)
 *   4. Exploit triangular routes through Pool C → Tier 5
 */
contract Arbitrage {
    address public owner;
    address public flashLender;
    address[] public pools;
    address public weth;
    address public usdc;
    address public dai;
    
    uint256 public initialBalance;
    uint256 public finalBalance;
    
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
        initialBalance = IERC20(weth).balanceOf(address(this));
        
        IDEXPool poolA = IDEXPool(pools[0]);  // Cheap WETH (1700)
        IDEXPool poolB = IDEXPool(pools[1]);  // Expensive WETH (2000)
        
        // SUBOPTIMAL: Only trade 2 WETH once (leaving 8 WETH unused!)
        uint256 tradeAmount = 2e18;  // 2 WETH
        
        // Get Pool B reserves
        (uint112 r0B, uint112 r1B) = poolB.getReserves();
        
        // Sell WETH in Pool B (where it's expensive)
        uint256 usdcOut = _getAmountOut(tradeAmount, r0B, r1B);
        IERC20(weth).transfer(address(poolB), tradeAmount);
        poolB.swap(0, usdcOut, address(this));
        
        // Buy WETH in Pool A (where it's cheap)
        (uint112 r0A, uint112 r1A) = poolA.getReserves();
        uint256 usdcBal = IERC20(usdc).balanceOf(address(this));
        uint256 wethOut = _getAmountOut(usdcBal, r1A, r0A);
        IERC20(usdc).transfer(address(poolA), usdcBal);
        poolA.swap(wethOut, 0, address(this));
        
        finalBalance = IERC20(weth).balanceOf(address(this));
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
        return finalBalance > initialBalance ? finalBalance - initialBalance : 0;
    }
    
    // Required for flash loan callback
    function onFlashLoan(
        address,
        address,
        uint256,
        uint256,
        bytes calldata
    ) external pure returns (bytes32) {
        return keccak256("ERC3156FlashBorrower.onFlashLoan");
    }
    
    receive() external payable {}
}
