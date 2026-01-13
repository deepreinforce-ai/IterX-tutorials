// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IERC3156FlashBorrower {
    function onFlashLoan(
        address initiator,
        address token,
        uint256 amount,
        uint256 fee,
        bytes calldata data
    ) external returns (bytes32);
}

/**
 * @title MockFlashLender
 * @notice EIP-3156 compatible flash loan provider for testing MEV arbitrage
 * @dev Provides flash loans with configurable fee
 */
contract MockFlashLender {
    bytes32 public constant CALLBACK_SUCCESS = keccak256("ERC3156FlashBorrower.onFlashLoan");
    
    mapping(address => bool) public supportedTokens;
    mapping(address => uint256) public maxFlashLoan;
    uint256 public feeBasisPoints;  // Fee in basis points (e.g., 9 = 0.09%)
    
    event FlashLoan(
        address indexed borrower,
        address indexed token,
        uint256 amount,
        uint256 fee
    );
    
    constructor(uint256 _feeBasisPoints) {
        feeBasisPoints = _feeBasisPoints;
    }
    
    /**
     * @notice Add a token that can be flash loaned
     * @param token Token address
     * @param maxAmount Maximum amount that can be borrowed
     */
    function addSupportedToken(address token, uint256 maxAmount) external {
        supportedTokens[token] = true;
        maxFlashLoan[token] = maxAmount;
    }
    
    /**
     * @notice Deposit tokens to be available for flash loans
     * @param token Token to deposit
     * @param amount Amount to deposit
     */
    function deposit(address token, uint256 amount) external {
        IERC20(token).transferFrom(msg.sender, address(this), amount);
        if (maxFlashLoan[token] < amount) {
            maxFlashLoan[token] = amount;
        }
    }
    
    /**
     * @notice Get maximum flash loan amount for a token
     * @param token Token address
     * @return Maximum amount available
     */
    function maxFlashLoanAmount(address token) external view returns (uint256) {
        if (!supportedTokens[token]) return 0;
        return IERC20(token).balanceOf(address(this));
    }
    
    /**
     * @notice Calculate flash loan fee
     * @param token Token address (unused, same fee for all)
     * @param amount Loan amount
     * @return Fee amount
     */
    function flashFee(address token, uint256 amount) public view returns (uint256) {
        require(supportedTokens[token], "Unsupported token");
        return (amount * feeBasisPoints) / 10000;
    }
    
    /**
     * @notice Execute a flash loan
     * @param borrower Address receiving the loan and callback
     * @param token Token to borrow
     * @param amount Amount to borrow
     * @param data Arbitrary data passed to callback
     */
    function flashLoan(
        address borrower,
        address token,
        uint256 amount,
        bytes calldata data
    ) external returns (bool) {
        require(supportedTokens[token], "Unsupported token");
        
        uint256 balanceBefore = IERC20(token).balanceOf(address(this));
        require(balanceBefore >= amount, "Insufficient liquidity");
        
        uint256 fee = flashFee(token, amount);
        
        // Transfer tokens to borrower
        IERC20(token).transfer(borrower, amount);
        
        // Call borrower's callback
        bytes32 result = IERC3156FlashBorrower(borrower).onFlashLoan(
            msg.sender,  // initiator
            token,
            amount,
            fee,
            data
        );
        require(result == CALLBACK_SUCCESS, "Callback failed");
        
        // Verify repayment
        uint256 balanceAfter = IERC20(token).balanceOf(address(this));
        require(balanceAfter >= balanceBefore + fee, "Flash loan not repaid");
        
        emit FlashLoan(borrower, token, amount, fee);
        
        return true;
    }
    
    /**
     * @notice Withdraw tokens (for testing cleanup)
     * @param token Token to withdraw
     * @param to Recipient
     * @param amount Amount to withdraw
     */
    function withdraw(address token, address to, uint256 amount) external {
        IERC20(token).transfer(to, amount);
    }
}






