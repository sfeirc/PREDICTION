"""
Custom loss functions for imbalanced classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of modulating factor (1 - p_t) to focus on hard examples
        reduction: 'mean', 'sum' or 'none'
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) raw logits
            targets: (batch_size,) class indices
        
        Returns:
            loss: scalar tensor
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balancing
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    
    Instead of hard targets [0, 1], use soft targets [ε/(C-1), 1-ε]
    where ε is the smoothing parameter and C is number of classes.
    
    This prevents overconfidence and improves generalization.
    
    Args:
        smoothing: Label smoothing parameter (default: 0.1)
        reduction: 'mean', 'sum' or 'none'
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) raw logits
            targets: (batch_size,) class indices
        
        Returns:
            loss: scalar tensor
        """
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        
        # Create smooth labels
        with torch.no_grad():
            # Start with uniform distribution
            smooth_targets = torch.zeros_like(log_preds)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            
            # Set true class probability
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute KL divergence
        loss = -smooth_targets * log_preds
        
        if self.reduction == 'mean':
            return loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=-1)


def get_loss_function(config: dict) -> nn.Module:
    """
    Get loss function based on config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Loss function module
    """
    loss_type = config["training"].get("loss_type", "cross_entropy")
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    elif loss_type == "focal":
        alpha = config["training"].get("focal_alpha", 0.25)
        gamma = config["training"].get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == "label_smoothing":
        smoothing = config["training"].get("label_smoothing", 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

