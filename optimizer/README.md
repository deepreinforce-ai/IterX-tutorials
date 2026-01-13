# Task: Design an Optimizer Better Than Adam
**Difficulty Level: ⭐⭐⭐⭐ (4 stars)**

---

## Background

Our goal is to find an optimization algorithm that performs better than Adam, meaning it can reach optimal performance faster.

To do this, we first fix both the dataset and the model. Then, we use IterX to update the optimization algorithm.

For illustration purposes, we choose **CIFAR-10** as the dataset and **ResNet-18** as the model. We also need to design a reward function to measure each optimizer variant.

---

## Reward Mechanism

The evaluation metric is designed to capture both optimization efficiency and final performance. Concretely, we compute:

1. **Inverted AUC of the training loss curve**, which measures how quickly the optimizer converges (higher is better).
2. **Final test accuracy**, which reflects the quality of the converged solution.

The overall reward score is defined as a weighted combination of these two metrics:

$$
\text{Reward} = \frac{1}{3} \cdot \text{Inverted AUC} + \frac{2}{3} \cdot \text{Final Test Accuracy}
$$

```python
def compute_reward(metrics):
    """
    Compute the overall reward score.
    
    Reward = (1/3) * Inverted_AUC + (2/3) * Final_Test_Accuracy
    
    Args:
        metrics: List of epoch metrics containing train_loss and test_acc
    
    Returns:
        reward: Combined score (higher is better, range [0, 1])
    """
    # Extract training losses
    train_losses = [m['train_loss'] for m in metrics if m['train_loss'] is not None]
    
    # Compute inverted AUC (faster convergence = higher score)
    inverted_auc = compute_inverted_auc(train_losses)
    
    # Get final test accuracy
    final_test_acc = metrics[-1]['test_acc']
    
    # Compute weighted reward
    reward = (1/3) * inverted_auc + (2/3) * final_test_acc
    
    return reward
```

---

## Task Description

**Goal:** Design an optimization algorithm that outperforms Adam on CIFAR-10 with ResNet-18.

**Interface:**
- Your optimizer must be a Python class with `zero_grad()` and `step()` methods
- Constructor should accept `params` (model parameters) and `lr` (learning rate)

**Requirements:**
1. Must implement standard PyTorch optimizer interface
2. Must handle gradient updates without causing NaN/Inf values
3. Should work with default learning rate of 0.001

**Things to Avoid:**
- Excessive memory usage
- Non-finite gradient values

---

## Reward Description

The `get_reward()` function returns a tuple of **`(reward, error_msg, details)`**:
- **`reward`**: Combined score in range `[0.0, 1.0]`.
- **`error_msg`**: Empty string `""` on success, or a description of the failure.
- **`details`**: Always `""`.

The reward combines convergence speed and final accuracy:

```
reward = (1/3) * inverted_auc + (2/3) * final_test_accuracy
```

- **Higher reward = better** (max = 1.0)
- **Inverted AUC** rewards optimizers that reduce training loss quickly
- **Final test accuracy** rewards optimizers that achieve good generalization

---

## Initial Code

We can simply use Adam as the initial code:

```python
import torch
import math

class CustomOptimizer(torch.optim.Optimizer):
    """Adam optimizer implementation."""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
```

---

## File Structure

```
optimizer/
├── README.md                 # This file
├── eval_optimizer.py         # Evaluation script
├── resnet.py                 # ResNet model definitions
├── initial_code.py           # Initial Adam optimizer
└── run_iterx.py              # IterX evaluation runner
```

---

## Running Evaluation

```bash
python eval_optimizer.py --task resnet18 --epoch 20 --device 0
```

Or programmatically:

```python
from eval_optimizer import get_reward

# Get reward for an optimizer implementation
reward = get_reward('path/to/your/optimizer.py')
print(reward)  # e.g., 0.65 (higher = better, max = 1.0)
```

---

## Running iterX

```bash
python run_iterx.py
```

