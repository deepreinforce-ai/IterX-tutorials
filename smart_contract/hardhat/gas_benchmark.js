const { ethers } = require("hardhat");

async function main() {
  // Deploy MockERC20
  const MockERC20 = await ethers.getContractFactory("MockERC20");
  const token = await MockERC20.deploy("Test Token", "TEST", 18);
  await token.waitForDeployment();

  // Deploy BatchTransfer
  const BatchTransfer = await ethers.getContractFactory("BatchTransfer");
  const batchTransfer = await BatchTransfer.deploy();
  await batchTransfer.waitForDeployment();

  const [owner] = await ethers.getSigners();
  
  const batchSizes = [10, 50, 100, 500];
  const results = {};

  for (const batchSize of batchSizes) {
    const recipients = [];
    const amounts = [];
    
    for (let i = 0; i < batchSize; i++) {
      const addr = ethers.Wallet.createRandom().address;
      recipients.push(addr);
      amounts.push(ethers.parseEther("1"));
    }

    const totalAmount = ethers.parseEther(String(batchSize));
    await token.mint(owner.address, totalAmount);
    await token.approve(await batchTransfer.getAddress(), totalAmount);

    const tx = await batchTransfer.batchTransfer(
      await token.getAddress(),
      recipients,
      amounts
    );
    const receipt = await tx.wait();
    
    results[batchSize] = Number(receipt.gasUsed);
  }

  console.log("GAS_RESULT:" + JSON.stringify(results));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
