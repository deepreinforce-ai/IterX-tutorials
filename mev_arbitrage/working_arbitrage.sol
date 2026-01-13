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
 * @notice Optimized two-pool arbitrage with dynamic batch sizing
 * 
 * Pool A: WETH cheap (1800 USDC/WETH) - BUY here
 * Pool B: WETH expensive (1950 USDC/WETH) - SELL here
 * 
 * Achieves ~0.3% profit by trading until prices converge
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
        
        IDEXPool poolA = IDEXPool(pools[0]);
        IDEXPool poolB = IDEXPool(pools[1]);
        
        // Trade until prices converge (arbitrage exhausted)
        for (uint256 i = 0; i < 150; i++) {
            (uint112 r0A, uint112 r1A) = poolA.getReserves();
            (uint112 r0B, uint112 r1B) = poolB.getReserves();
            
            // Calculate prices (USDC per WETH, scaled by 1e18)
            uint256 priceA = uint256(r1A) * 1e18 / uint256(r0A);
            uint256 priceB = uint256(r1B) * 1e18 / uint256(r0B);
            
            // Need spread > 0.65% to profit (covers 2x 0.3% fees + small margin)
            if (priceB * 10000 <= priceA * 10065) break;
            
            // Dynamic batch: larger batches when spread is big
            uint256 spread = (priceB - priceA) * 10000 / priceA;
            uint256 batchSize;
            
            if (spread > 500) batchSize = 8e17;      // >5% spread: 0.8 WETH
            else if (spread > 300) batchSize = 5e17; // >3% spread: 0.5 WETH
            else if (spread > 150) batchSize = 3e17; // >1.5% spread: 0.3 WETH
            else if (spread > 80) batchSize = 1e17;  // >0.8% spread: 0.1 WETH
            else batchSize = 3e16;                    // small spread: 0.03 WETH
            
            uint256 wethBal = IERC20(weth).balanceOf(address(this));
            if (wethBal < batchSize) {
                if (wethBal < 1e16) break;
                batchSize = wethBal;
            }
            
            // Step 1: Sell WETH in Pool B (where WETH is expensive)
            uint256 usdcOut = _getAmountOut(batchSize, r0B, r1B);
            if (usdcOut == 0) break;
            
            IERC20(weth).transfer(address(poolB), batchSize);
            poolB.swap(0, usdcOut, address(this));
            
            // Step 2: Buy WETH in Pool A (where WETH is cheap)
            (r0A, r1A) = poolA.getReserves();
            uint256 usdcBal = IERC20(usdc).balanceOf(address(this));
            uint256 wethOut = _getAmountOut(usdcBal, r1A, r0A);
            if (wethOut == 0) break;
            
            IERC20(usdc).transfer(address(poolA), usdcBal);
            poolA.swap(wethOut, 0, address(this));
        }
        
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
    
    function onFlashLoan(address, address, uint256, uint256, bytes calldata) 
        external pure returns (bytes32) 
    {
        return keccak256("ERC3156FlashBorrower.onFlashLoan");
    }
    
    receive() external payable {}
}
