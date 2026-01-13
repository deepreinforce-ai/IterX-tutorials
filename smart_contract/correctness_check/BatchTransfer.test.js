const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("BatchTransfer", function () {
  let token;
  let batchTransfer;
  let owner;
  let recipients;

  beforeEach(async function () {
    [owner, ...recipients] = await ethers.getSigners();

    const MockERC20 = await ethers.getContractFactory("MockERC20");
    token = await MockERC20.deploy("Test Token", "TEST", 18);
    await token.waitForDeployment();

    const BatchTransfer = await ethers.getContractFactory("BatchTransfer");
    batchTransfer = await BatchTransfer.deploy();
    await batchTransfer.waitForDeployment();
  });

  it("should transfer tokens to multiple recipients", async function () {
    const recipientAddresses = recipients.slice(0, 3).map(r => r.address);
    const amounts = [
      ethers.parseEther("100"),
      ethers.parseEther("200"),
      ethers.parseEther("300")
    ];
    const totalAmount = ethers.parseEther("600");

    await token.mint(owner.address, totalAmount);
    await token.approve(await batchTransfer.getAddress(), totalAmount);

    await expect(
      batchTransfer.batchTransfer(await token.getAddress(), recipientAddresses, amounts)
    ).to.emit(batchTransfer, "BatchTransferCompleted");

    expect(await token.balanceOf(recipientAddresses[0])).to.equal(amounts[0]);
    expect(await token.balanceOf(recipientAddresses[1])).to.equal(amounts[1]);
    expect(await token.balanceOf(recipientAddresses[2])).to.equal(amounts[2]);
  });

  it("should revert if arrays have different lengths", async function () {
    const recipientAddresses = recipients.slice(0, 2).map(r => r.address);
    const amounts = [ethers.parseEther("100")];

    await token.mint(owner.address, ethers.parseEther("100"));
    await token.approve(await batchTransfer.getAddress(), ethers.parseEther("100"));

    await expect(
      batchTransfer.batchTransfer(await token.getAddress(), recipientAddresses, amounts)
    ).to.be.reverted;
  });

  it("should handle single transfer", async function () {
    const recipientAddresses = [recipients[0].address];
    const amounts = [ethers.parseEther("50")];

    await token.mint(owner.address, ethers.parseEther("50"));
    await token.approve(await batchTransfer.getAddress(), ethers.parseEther("50"));

    await batchTransfer.batchTransfer(await token.getAddress(), recipientAddresses, amounts);

    expect(await token.balanceOf(recipientAddresses[0])).to.equal(amounts[0]);
  });
});
