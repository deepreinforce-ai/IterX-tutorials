import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet import ResNet18, ResNet34
import importlib.util
import sys
import json
import traceback
import logging
from datetime import datetime
import time
import math

def setup_logging(log_level='INFO', log_file=None):
    """Setup logging with proper formatting"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr),  # Always log to stderr
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10')
    parser.add_argument('--device', type=int, default=0, help='Device to use for training')
    parser.add_argument('--epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for training and testing')
    parser.add_argument('--task', type=str, required=True, choices=['resnet18', 'resnet34'], 
                       help='ResNet architecture to use (resnet18 or resnet34)')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log_file', type=str, help='Optional log file path')
    parser.add_argument('--output_format', type=str, default='json', choices=['json', 'human'], 
                       help='Output format: json for machine parsing, human for readability')
    return parser.parse_args()

def load_optimizer_from_file(file_path, logger):
    """
    Dynamically load optimizer class with comprehensive error handling
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Optimizer file not found: {file_path}")
        
        logger.info(f"Loading optimizer from: {file_path}")
        
        # Create module spec from file path
        spec = importlib.util.spec_from_file_location("optimizer_module", file_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {file_path}")
        
        # Create module from spec
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module with error capture
        spec.loader.exec_module(module)
        
        # Find optimizer class
        optimizer_classes = []
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, 'zero_grad') and 
                hasattr(obj, 'step') and 
                name != 'Optimizer'):  # Exclude base Optimizer class
                optimizer_classes.append((name, obj))
        
        if not optimizer_classes:
            raise ImportError(f"No valid optimizer class found in {file_path}")
        
        if len(optimizer_classes) > 1:
            logger.warning(f"Multiple optimizer classes found: {[name for name, _ in optimizer_classes]}. Using first one.")
        
        optimizer_name, optimizer_class = optimizer_classes[0]
        logger.info(f"Successfully loaded optimizer class: {optimizer_name}")
        
        return optimizer_class, optimizer_name
        
    except Exception as e:
        logger.error(f"Failed to load optimizer: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def get_data_loaders(batch_size, logger):
    """Load CIFAR-10 data with error handling"""
    try:
        logger.info(f"Loading CIFAR-10 data with batch size: {batch_size}")
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Use more flexible data path
        data_root = os.environ.get('CIFAR_DATA_ROOT', './data')
        
        trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

        logger.info(f"Data loaded successfully. Train samples: {len(trainset)}, Test samples: {len(testset)}")
        return trainloader, testloader
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def setup_model(learning_rate, optimizer_class, device_index, task_name, logger):
    """Setup model with comprehensive error handling"""
    try:
        if torch.cuda.is_available() and device_index >= 0:
            device = torch.device(f"cuda:{device_index}")
            logger.info(f"Using CUDA device: {device}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
            
        # Choose the appropriate ResNet model
        if task_name == 'resnet18':
            model = ResNet18().to(device)
            logger.info("Using ResNet18 architecture")
        elif task_name == 'resnet34':
            model = ResNet34().to(device)
            logger.info("Using ResNet34 architecture")
        else:
            raise ValueError(f"Unsupported task: {task_name}")
            
        criterion = nn.CrossEntropyLoss()
        
        # Try to create optimizer with error handling
        try:
            optimizer = optimizer_class(model.parameters(), lr=learning_rate)
            logger.info(f"Optimizer created successfully with lr={learning_rate}")
        except Exception as e:
            logger.error(f"Failed to create optimizer: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
        return model, criterion, optimizer, device
        
    except Exception as e:
        logger.error(f"Failed to setup model: {str(e)}")
        raise

def train_epoch(epoch, trainloader, model, criterion, optimizer, device, logger):
    """Train one epoch with detailed error handling"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Check for NaN/Inf in gradients
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if not torch.isfinite(torch.tensor(grad_norm)):
            logger.warning(f"Non-finite gradient norm detected at batch {batch_idx}")
            continue
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
            
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(trainloader) if len(trainloader) > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"Epoch {epoch} training completed in {epoch_time:.2f}s - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    
    return avg_loss, accuracy
        

def test_epoch(epoch, testloader, model, criterion, device, logger):
    """Test one epoch with error handling"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            try:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            except Exception as e:
                logger.warning(f"Error in test batch {batch_idx}: {str(e)}")
                continue

    test_loss /= len(testloader) if len(testloader) > 0 else 1
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"Epoch {epoch} testing - Loss: {test_loss:.4f}, Acc: {accuracy:.4f}")
    
    return test_loss, accuracy
        

def compute_inverted_auc(train_losses):
    """
    Compute inverted AUC of training loss curve.
    
    Inverted AUC measures how quickly the optimizer converges.
    Higher is better (faster convergence = smaller area under loss curve).
    
    Args:
        train_losses: List of training losses per epoch
    
    Returns:
        Inverted AUC normalized to [0, 1]
    """
    if not train_losses or len(train_losses) < 2:
        return 0.0
    
    # Filter out None values
    losses = [l for l in train_losses if l is not None]
    if len(losses) < 2:
        return 0.0
    
    # Normalize losses to [0, 1] range
    max_loss = max(losses)
    min_loss = min(losses)
    
    if max_loss == min_loss:
        return 0.5  # All losses are the same
    
    normalized_losses = [(l - min_loss) / (max_loss - min_loss) for l in losses]
    
    # Compute AUC using trapezoidal rule
    n = len(normalized_losses)
    auc = sum((normalized_losses[i] + normalized_losses[i+1]) / 2 for i in range(n-1)) / (n-1)
    
    # Invert: higher is better (1 - AUC)
    inverted_auc = 1.0 - auc
    
    return inverted_auc


def compute_reward(metrics):
    """
    Compute the overall reward score.
    
    Reward = (1/3) * Inverted_AUC + (2/3) * Final_Test_Accuracy
    
    Args:
        metrics: List of epoch metrics containing train_loss and test_acc
    
    Returns:
        reward: Combined score (higher is better)
    """
    if not metrics:
        return 0.0
    
    # Extract training losses (excluding epoch 0 which has None)
    train_losses = [m['train_loss'] for m in metrics if m['train_loss'] is not None]
    
    # Compute inverted AUC
    inverted_auc = compute_inverted_auc(train_losses)
    
    # Get final test accuracy (already in [0, 1])
    final_test_acc = metrics[-1]['test_acc'] if metrics else 0.0
    
    # Compute weighted reward
    reward = (1/3) * inverted_auc + (2/3) * final_test_acc
    
    return reward


def output_result(result, output_format):
    """Output result in specified format"""
    if output_format == 'json':
        # JSON output goes to stdout for machine parsing
        print(json.dumps(result, indent=None, separators=(',', ':')))
    else:
        # Human-readable output
        if result['success']:
            print(f"Training completed successfully!")
            print(f"Optimizer: {result['optimizer_name']}")
            print(f"Reward: {result['reward']:.4f}")
            print(f"Total epochs: {len(result['metrics'])}")
            for metric in result['metrics']:
                if metric['train_loss'] is not None:
                    print(f"Epoch {metric['epoch']}: Train Loss={metric['train_loss']:.4f}, "
                          f"Train Acc={metric['train_acc']:.4f}, Test Loss={metric['test_loss']:.4f}, "
                          f"Test Acc={metric['test_acc']:.4f}")
        else:
            print(f"Training failed: {result['error']}")


def get_reward(code: str):
    """
    Evaluate an optimizer and return the reward.
    
    Args:
        code: Python code string containing the optimizer class
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: float in range [0.0, 1.0]
        - error_msg: "" if successful, error description otherwise
        - details: "" (reserved for future use)
    """
    import tempfile
    
    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        optimizer_path = f.name
    
    try:
        args = parse_args()
        
        # Setup logging
        logger = setup_logging(args.log_level, args.log_file)
        
        start_time = datetime.now()
        logger.info(f"Starting training at {start_time}")
        logger.info(f"Arguments: {vars(args)}")
        
        result = {
            'success': False,
            'error': None,
            'optimizer_name': None,
            'metrics': [],
            'start_time': start_time.isoformat(),
            'end_time': None,
            'duration_seconds': None
        }
        
        error_msg = ""
        
        try:
            # Load optimizer
            optimizer_class, optimizer_name = load_optimizer_from_file(optimizer_path, logger)
        result['optimizer_name'] = optimizer_name
        
        # Load data
        trainloader, testloader = get_data_loaders(args.batch, logger)
        
        # Setup model
        model, criterion, optimizer, device = setup_model(0.001, optimizer_class, args.device, args.task, logger)
        
        # Training loop
        epoch_metrics = []
        
        # Initial evaluation before training (epoch 0)
        logger.info("Evaluating initial model performance (epoch 0)")
        # initial_test_loss, initial_test_acc = test_epoch(0, testloader, model, criterion, device, logger)
        epoch_metrics.append({
            'epoch': 0,
            'train_loss': None,  # No training performed yet
            'train_acc': None,   # No training performed yet
            'test_loss': float(-math.log(0.1)),
            'test_acc': float(0.1)
        })
        
        for epoch in range(1, args.epoch + 1):
            train_loss, train_acc = train_epoch(epoch, trainloader, model, criterion, optimizer, device, logger)
            test_loss, test_acc = test_epoch(epoch, testloader, model, criterion, device, logger)
            
            # Store metrics
            epoch_metrics.append({
                'epoch': epoch,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'test_loss': float(test_loss),
                'test_acc': float(test_acc)
            })
        
        result['metrics'] = epoch_metrics
        result['success'] = True
        logger.info(f"Training completed successfully with {len(epoch_metrics)} epochs")
        
    except Exception as e:
        error_msg = str(e)
        result['error'] = error_msg
        logger.error(f"Training failed: {error_msg}")
        logger.debug(traceback.format_exc())
    
    finally:
        end_time = datetime.now()
        result['end_time'] = end_time.isoformat()
        result['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Compute reward
        reward = compute_reward(result['metrics']) if result['success'] else 0.0
        result['reward'] = reward
        
        logger.info(f"Reward: {reward:.4f}")
        
        # Output final result
        output_result(result, args.output_format)
        
        return reward, error_msg, ""
    finally:
        os.unlink(optimizer_path)

if __name__ == '__main__':
    with open(os.path.join(os.getcwd(), "initial_code.py")) as f:
        code = f.read()
    reward, error_msg, details = get_reward(code)
    print(f"Reward: {reward}, Error: {error_msg}")
