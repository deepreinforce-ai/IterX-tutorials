// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

/**
 * @title MockDEXPool
 * @notice UniswapV2-style constant product AMM pool for testing MEV arbitrage
 * @dev Implements x * y = k formula with 0.3% swap fee
 */
contract MockDEXPool {
    address public token0;
    address public token1;
    
    uint112 public reserve0;
    uint112 public reserve1;
    uint32 public blockTimestampLast;
    
    uint256 public constant MINIMUM_LIQUIDITY = 1000;
    
    event Sync(uint112 reserve0, uint112 reserve1);
    event Swap(
        address indexed sender,
        uint amount0In,
        uint amount1In,
        uint amount0Out,
        uint amount1Out,
        address indexed to
    );
    
    constructor(address _token0, address _token1) {
        token0 = _token0;
        token1 = _token1;
    }
    
    /**
     * @notice Get current reserves
     */
    function getReserves() external view returns (uint112 _reserve0, uint112 _reserve1) {
        _reserve0 = reserve0;
        _reserve1 = reserve1;
    }
    
    /**
     * @notice Add initial liquidity to the pool
     * @dev Only for setup - tokens must be transferred to pool first
     */
    function addLiquidity(uint112 amount0, uint112 amount1) external {
        // Transfer tokens to pool
        IERC20(token0).transferFrom(msg.sender, address(this), amount0);
        IERC20(token1).transferFrom(msg.sender, address(this), amount1);
        
        reserve0 = amount0;
        reserve1 = amount1;
        blockTimestampLast = uint32(block.timestamp);
        
        emit Sync(reserve0, reserve1);
    }
    
    /**
     * @notice Set reserves directly (for testing setup)
     * @dev Tokens must already be in the pool
     */
    function setReserves(uint112 _reserve0, uint112 _reserve1) external {
        reserve0 = _reserve0;
        reserve1 = _reserve1;
        emit Sync(reserve0, reserve1);
    }
    
    /**
     * @notice Swap tokens using constant product formula
     * @param amount0Out Amount of token0 to receive
     * @param amount1Out Amount of token1 to receive
     * @param to Recipient address
     */
    function swap(uint amount0Out, uint amount1Out, address to) external {
        require(amount0Out > 0 || amount1Out > 0, "INSUFFICIENT_OUTPUT_AMOUNT");
        require(amount0Out < reserve0 && amount1Out < reserve1, "INSUFFICIENT_LIQUIDITY");
        require(to != token0 && to != token1, "INVALID_TO");
        
        // Transfer out tokens
        if (amount0Out > 0) IERC20(token0).transfer(to, amount0Out);
        if (amount1Out > 0) IERC20(token1).transfer(to, amount1Out);
        
        // Get new balances
        uint balance0 = IERC20(token0).balanceOf(address(this));
        uint balance1 = IERC20(token1).balanceOf(address(this));
        
        // Calculate amounts in
        uint amount0In = balance0 > reserve0 - amount0Out ? balance0 - (reserve0 - amount0Out) : 0;
        uint amount1In = balance1 > reserve1 - amount1Out ? balance1 - (reserve1 - amount1Out) : 0;
        require(amount0In > 0 || amount1In > 0, "INSUFFICIENT_INPUT_AMOUNT");
        
        // Verify k invariant with 0.3% fee
        // (balance0 * 1000 - amount0In * 3) * (balance1 * 1000 - amount1In * 3) >= reserve0 * reserve1 * 1000000
        uint balance0Adjusted = balance0 * 1000 - amount0In * 3;
        uint balance1Adjusted = balance1 * 1000 - amount1In * 3;
        require(
            balance0Adjusted * balance1Adjusted >= uint(reserve0) * uint(reserve1) * 1000000,
            "K_INVARIANT"
        );
        
        // Update reserves
        reserve0 = uint112(balance0);
        reserve1 = uint112(balance1);
        blockTimestampLast = uint32(block.timestamp);
        
        emit Swap(msg.sender, amount0In, amount1In, amount0Out, amount1Out, to);
        emit Sync(reserve0, reserve1);
    }
    
    /**
     * @notice Calculate output amount for a given input
     * @param amountIn Input token amount
     * @param reserveIn Reserve of input token
     * @param reserveOut Reserve of output token
     * @return amountOut Output token amount
     */
    function getAmountOut(uint amountIn, uint reserveIn, uint reserveOut) 
        external pure returns (uint amountOut) 
    {
        require(amountIn > 0, "INSUFFICIENT_INPUT_AMOUNT");
        require(reserveIn > 0 && reserveOut > 0, "INSUFFICIENT_LIQUIDITY");
        
        uint amountInWithFee = amountIn * 997;  // 0.3% fee
        uint numerator = amountInWithFee * reserveOut;
        uint denominator = (reserveIn * 1000) + amountInWithFee;
        amountOut = numerator / denominator;
    }
    
    /**
     * @notice Calculate input amount for a desired output
     * @param amountOut Desired output amount
     * @param reserveIn Reserve of input token
     * @param reserveOut Reserve of output token
     * @return amountIn Required input amount
     */
    function getAmountIn(uint amountOut, uint reserveIn, uint reserveOut)
        external pure returns (uint amountIn)
    {
        require(amountOut > 0, "INSUFFICIENT_OUTPUT_AMOUNT");
        require(reserveIn > 0 && reserveOut > 0, "INSUFFICIENT_LIQUIDITY");
        
        uint numerator = reserveIn * amountOut * 1000;
        uint denominator = (reserveOut - amountOut) * 997;
        amountIn = (numerator / denominator) + 1;
    }
    
    /**
     * @notice Sync reserves to current balances
     */
    function sync() external {
        reserve0 = uint112(IERC20(token0).balanceOf(address(this)));
        reserve1 = uint112(IERC20(token1).balanceOf(address(this)));
        emit Sync(reserve0, reserve1);
    }
}






