const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("MEV Arbitrage Evaluation", function () {
    let weth, usdc, dai;
    let poolA, poolB, poolC, poolD;
    let flashLender;
    let arbitrage;
    let deployer;

    // ============================================================================
    // POOL CONFIGURATION - Designed for Progressive Optimization
    // ============================================================================
    // 
    // Strategy tiers (expected profit ranges):
    //   Tier 0: Compilation only, no profit (reward ~0)
    //   Tier 1: Basic arbitrage, single/few trades (0.1-0.5 WETH profit)
    //   Tier 2: Batched trades with own capital (0.5-1.5 WETH profit)
    //   Tier 3: Optimal batching (1.5-3.0 WETH profit)
    //   Tier 4: Flash loan amplified (3.0-5.0 WETH profit)
    //   Tier 5: Optimal flash + multi-route (5.0+ WETH profit)
    //
    // ============================================================================

    // Pool A: WETH/USDC - CHEAP WETH (price: 1700 USDC/WETH)
    const POOL_A_WETH = ethers.parseEther("5000");          // 5000 WETH (huge!)
    const POOL_A_USDC = ethers.parseUnits("8500000", 6);    // 8,500,000 USDC (price: 1700)
    
    // Pool B: WETH/USDC - EXPENSIVE WETH (price: 2000 USDC/WETH) - 17.6% spread!
    const POOL_B_WETH = ethers.parseEther("4000");          // 4000 WETH (huge!)
    const POOL_B_USDC = ethers.parseUnits("8000000", 6);    // 8,000,000 USDC (price: 2000)
    
    // Pool C: WETH/DAI - MID PRICE (price: 1850 DAI/WETH) - triangular opportunity
    const POOL_C_WETH = ethers.parseEther("3000");          // 3000 WETH (huge!)
    const POOL_C_DAI = ethers.parseEther("5550000");        // 5,550,000 DAI (price: 1850)
    
    // Pool D: USDC/DAI - MISPRICED STABLES (1 USDC = 1.02 DAI) - enables triangular
    const POOL_D_USDC = ethers.parseUnits("5000000", 6);    // 5,000,000 USDC
    const POOL_D_DAI = ethers.parseEther("5100000");        // 5,100,000 DAI (slight misprice)
    
    const FLASH_LOAN_LIQUIDITY = ethers.parseEther("10000"); // 10000 WETH for flash loans
    const INITIAL_CAPITAL = ethers.parseEther("10");        // 10 WETH starting capital
    
    // Flash loan fee: 0.05% (5 basis points) - low enough to be profitable
    const FLASH_FEE_BPS = 5;

    beforeEach(async function () {
        [deployer] = await ethers.getSigners();

        // Deploy tokens
        const MockERC20 = await ethers.getContractFactory("MockERC20");
        weth = await MockERC20.deploy("Wrapped Ether", "WETH", 18);
        usdc = await MockERC20.deploy("USD Coin", "USDC", 6);
        dai = await MockERC20.deploy("Dai Stablecoin", "DAI", 18);

        // Deploy DEX pools
        const MockDEXPool = await ethers.getContractFactory("MockDEXPool");
        poolA = await MockDEXPool.deploy(await weth.getAddress(), await usdc.getAddress());
        poolB = await MockDEXPool.deploy(await weth.getAddress(), await usdc.getAddress());
        poolC = await MockDEXPool.deploy(await weth.getAddress(), await dai.getAddress());
        poolD = await MockDEXPool.deploy(await usdc.getAddress(), await dai.getAddress());

        // Deploy flash lender with lower fee (0.05%)
        const MockFlashLender = await ethers.getContractFactory("MockFlashLender");
        flashLender = await MockFlashLender.deploy(FLASH_FEE_BPS);

        // Mint tokens for pool liquidity
        await weth.mint(deployer.address, ethers.parseEther("100000"));
        await usdc.mint(deployer.address, ethers.parseUnits("100000000", 6));
        await dai.mint(deployer.address, ethers.parseEther("100000000"));

        // Approve pools
        await weth.approve(await poolA.getAddress(), ethers.MaxUint256);
        await usdc.approve(await poolA.getAddress(), ethers.MaxUint256);
        await weth.approve(await poolB.getAddress(), ethers.MaxUint256);
        await usdc.approve(await poolB.getAddress(), ethers.MaxUint256);
        await weth.approve(await poolC.getAddress(), ethers.MaxUint256);
        await dai.approve(await poolC.getAddress(), ethers.MaxUint256);
        await usdc.approve(await poolD.getAddress(), ethers.MaxUint256);
        await dai.approve(await poolD.getAddress(), ethers.MaxUint256);

        // Add liquidity to pools
        await poolA.addLiquidity(POOL_A_WETH, POOL_A_USDC);
        await poolB.addLiquidity(POOL_B_WETH, POOL_B_USDC);
        await poolC.addLiquidity(POOL_C_WETH, POOL_C_DAI);
        await poolD.addLiquidity(POOL_D_USDC, POOL_D_DAI);

        // Setup flash lender
        await flashLender.addSupportedToken(await weth.getAddress(), FLASH_LOAN_LIQUIDITY);
        await flashLender.addSupportedToken(await usdc.getAddress(), ethers.parseUnits("5000000", 6));
        await flashLender.addSupportedToken(await dai.getAddress(), ethers.parseEther("5000000"));
        
        await weth.approve(await flashLender.getAddress(), ethers.MaxUint256);
        await flashLender.deposit(await weth.getAddress(), FLASH_LOAN_LIQUIDITY);

        // Deploy arbitrage contract
        const Arbitrage = await ethers.getContractFactory("Arbitrage");
        arbitrage = await Arbitrage.deploy(
            await flashLender.getAddress(),
            [
                await poolA.getAddress(),
                await poolB.getAddress(),
                await poolC.getAddress(),
                await poolD.getAddress()
            ],
            [
                await weth.getAddress(),
                await usdc.getAddress(),
                await dai.getAddress()
            ]
        );

        // Give arbitrage contract starting capital
        await weth.mint(await arbitrage.getAddress(), INITIAL_CAPITAL);
    });

    it("Arbitrage - measures profit extraction", async function () {
        // Get initial balance
        const initialBalance = await weth.balanceOf(await arbitrage.getAddress());
        console.log(`Initial Balance: ${ethers.formatEther(initialBalance)} WETH`);

        // Log pool prices before
        const [r0A, r1A] = await poolA.getReserves();
        const [r0B, r1B] = await poolB.getReserves();
        const [r0C, r1C] = await poolC.getReserves();
        const [r0D, r1D] = await poolD.getReserves();
        
        const priceABefore = Number(r1A) / Number(r0A) * 1e12;
        const priceBBefore = Number(r1B) / Number(r0B) * 1e12;
        const priceCBefore = Number(r1C) / Number(r0C);
        const priceDai = Number(r1D) / Number(r0D) * 1e12;
        
        console.log(`Pool A price: ${priceABefore} USDC/WETH (CHEAP - BUY WETH HERE)`);
        console.log(`Pool B price: ${priceBBefore} USDC/WETH (EXPENSIVE - SELL WETH HERE)`);
        console.log(`Pool C price: ${priceCBefore} DAI/WETH`);
        console.log(`Pool D rate: ${priceDai} DAI/USDC`);
        
        const spreadAB = ((priceBBefore - priceABefore) / priceABefore * 100);
        console.log(`Arbitrage spread: ${spreadAB.toFixed(2)}%`);

        // Execute arbitrage
        let executionError = null;
        try {
            await arbitrage.executeArbitrage();
        } catch (error) {
            executionError = error.message;
            const match = error.message.match(/reverted with reason string '([^']+)'/);
            if (match) {
                executionError = match[1];
            }
            console.log(`Execution Error: ${executionError}`);
        }

        // Get final balance
        const finalBalance = await weth.balanceOf(await arbitrage.getAddress());
        console.log(`Final Balance: ${ethers.formatEther(finalBalance)} WETH`);

        // Calculate profit/loss
        const initialBN = BigInt(initialBalance);
        const finalBN = BigInt(finalBalance);
        let profitOrLoss;
        let isProfit = finalBN >= initialBN;
        
        if (isProfit) {
            profitOrLoss = finalBN - initialBN;
            console.log(`Profit: ${ethers.formatEther(profitOrLoss)} WETH`);
        } else {
            profitOrLoss = initialBN - finalBN;
            console.log(`Profit: 0.0 WETH`);
            console.log(`LOSS: ${ethers.formatEther(profitOrLoss)} WETH`);
            console.log(`Loss percentage: ${(Number(profitOrLoss) / Number(initialBN) * 100).toFixed(2)}%`);
        }

        // Log pool prices after
        const [r0AAfter, r1AAfter] = await poolA.getReserves();
        const [r0BAfter, r1BAfter] = await poolB.getReserves();
        const [r0CAfter, r1CAfter] = await poolC.getReserves();
        const priceAAfter = Number(r1AAfter) / Number(r0AAfter) * 1e12;
        const priceBAfter = Number(r1BAfter) / Number(r0BAfter) * 1e12;
        const priceCAfter = Number(r1CAfter) / Number(r0CAfter);
        
        console.log(`Pool A price after: ${priceAAfter} USDC/WETH`);
        console.log(`Pool B price after: ${priceBAfter} USDC/WETH`);
        console.log(`Pool C price after: ${priceCAfter} DAI/WETH`);

        // Calculate price impacts
        const priceImpactA = ((priceAAfter - priceABefore) / priceABefore * 100).toFixed(2);
        const priceImpactB = ((priceBAfter - priceBBefore) / priceBBefore * 100).toFixed(2);
        const priceImpactC = ((priceCAfter - priceCBefore) / priceCBefore * 100).toFixed(2);
        
        console.log(`Pool A price impact: ${priceImpactA}%`);
        console.log(`Pool B price impact: ${priceImpactB}%`);
        console.log(`Pool C price impact: ${priceImpactC}%`);

        // Hints for improvement
        if (!isProfit && !executionError) {
            const totalPriceImpact = Math.abs(parseFloat(priceImpactA)) + Math.abs(parseFloat(priceImpactB));
            if (totalPriceImpact > 10) {
                console.log(`HINT: Total price impact ${totalPriceImpact.toFixed(1)}% is too high. Use smaller batch sizes in a loop.`);
            }
            if (priceAAfter > priceBAfter) {
                console.log(`HINT: Pool prices CROSSED! Trade size way too large.`);
            }
        }

        if (executionError && executionError.includes('Insufficient balance')) {
            console.log(`HINT: Flash loan repayment failed. Use transfer() not approve() in onFlashLoan callback.`);
        }

        expect(finalBalance).to.be.gte(0);
    });
});
