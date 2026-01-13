// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IERC20.sol";

/**
 * @title BatchTransfer
 * @notice A contract for batch transferring ERC20 tokens to multiple recipients
 * @dev This is the BASELINE implementation - intentionally unoptimized
 * 
 * Gas Optimization Task:
 * Your goal is to reduce the gas consumption of the batchTransfer function
 * while maintaining the same functionality and security guarantees.
 * 
 * Current gas profile (approximate):
 * - 10 recipients: ~95,000 gas
 * - 50 recipients: ~425,000 gas
 * - 100 recipients: ~845,000 gas
 * - 500 recipients: ~4,200,000 gas
 */
contract BatchTransfer {
    
    /// @notice Emitted when a batch transfer is completed
    /// @param token The address of the token transferred
    /// @param sender The address that initiated the batch transfer
    /// @param totalAmount The total amount of tokens transferred
    /// @param recipientCount The number of recipients
    event BatchTransferCompleted(
        address indexed token,
        address indexed sender,
        uint256 totalAmount,
        uint256 recipientCount
    );

    /// @notice Emitted for each individual transfer within the batch
    /// @param recipient The address receiving tokens
    /// @param amount The amount transferred
    event IndividualTransfer(address indexed recipient, uint256 amount);

    /**
     * @notice Transfers tokens to multiple recipients in a single transaction
     * @param token The ERC20 token address to transfer
     * @param recipients Array of recipient addresses
     * @param amounts Array of amounts to transfer to each recipient
     * 
     * @dev Requirements:
     * - recipients and amounts arrays must have the same length
     * - Arrays must not be empty
     * - No recipient can be the zero address
     * - No amount can be zero
     * - Caller must have approved this contract for the total amount
     * - Token contract must return true on transferFrom
     */
    function batchTransfer(
        address token,
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external {
        // Input validation
        require(recipients.length == amounts.length, "BatchTransfer: length mismatch");
        require(recipients.length > 0, "BatchTransfer: empty arrays");
        
        // Cache the token contract reference
        IERC20 tokenContract = IERC20(token);
        
        // Track total amount for event
        uint256 totalAmount = 0;
        
        // Process each transfer
        for (uint256 i = 0; i < recipients.length; i++) {
            address recipient = recipients[i];
            uint256 amount = amounts[i];
            
            // Validate recipient
            require(recipient != address(0), "BatchTransfer: zero address recipient");
            
            // Validate amount
            require(amount > 0, "BatchTransfer: zero amount");
            
            // Update total
            totalAmount += amount;
            
            // Execute transfer
            bool success = tokenContract.transferFrom(msg.sender, recipient, amount);
            require(success, "BatchTransfer: transfer failed");
            
            // Emit individual transfer event (expensive but part of baseline)
            emit IndividualTransfer(recipient, amount);
        }
        
        // Emit completion event
        emit BatchTransferCompleted(token, msg.sender, totalAmount, recipients.length);
    }

    /**
     * @notice Helper function to calculate total amount needed for a batch
     * @param amounts Array of amounts
     * @return total The sum of all amounts
     */
    function calculateTotalAmount(uint256[] calldata amounts) external pure returns (uint256 total) {
        for (uint256 i = 0; i < amounts.length; i++) {
            total += amounts[i];
        }
    }

    /**
     * @notice Check if the caller has sufficient allowance for a batch transfer
     * @param token The ERC20 token address
     * @param owner The token owner address
     * @param amounts Array of amounts to transfer
     * @return sufficient True if allowance is sufficient
     * @return required The total amount required
     * @return current The current allowance
     */
    function checkAllowance(
        address token,
        address owner,
        uint256[] calldata amounts
    ) external view returns (bool sufficient, uint256 required, uint256 current) {
        for (uint256 i = 0; i < amounts.length; i++) {
            required += amounts[i];
        }
        current = IERC20(token).allowance(owner, address(this));
        sufficient = current >= required;
    }
}
