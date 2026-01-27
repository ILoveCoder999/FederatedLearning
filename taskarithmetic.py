
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

# ---------------------------------------------------------
# Part 1: SparseSGDM Optimizer 
# ---------------------------------------------------------
class SparseSGDM(optim.SGD):
    """
    Implements Stochastic Gradient Descent with Momentum (SGDM) 
    that supports gradient masking for sparse fine-tuning.
    
    Inherits from torch.optim.SGD.
    """
    def __init__(self, params, lr=0.001, momentum=0.9, weight_decay=0.0, masks=None):
        """
        :param params: Model parameters to optimize.
        :param masks: A dictionary mapping parameter names (or IDs) to binary masks.
                      If mask[i] == 0, the gradient for param[i] is zeroed out.
        """
        super(SparseSGDM, self).__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.masks = masks

    def set_masks(self, masks):
        """
        Update the masks used by the optimizer.
        """
        self.masks = masks

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over all parameter groups (standard PyTorch optimizer structure)
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    # -----------------------------------------------------------
                    # Key Modification for Task 3.3 
                    # "Zero-out the updates of the weights whose corresponding 
                    # entry in the mask is zero."
                    # -----------------------------------------------------------
                    if self.masks is not None:
                        # We identify the mask by the parameter tensor's object ID
                        # or we assume the `masks` passed is a list aligned with parameters.
                        # For simplicity in this implementation, we assume `self.masks`
                        # is a dictionary {param_tensor_id: mask_tensor} or we handle it externally.
                        
                        # However, a robust way for this project is to check if 
                        # the parameter has a state attribute for the mask.
                        pass 
                        
                        # PRACTICAL IMPLEMENTATION:
                        # Apply mask directly to p.grad before the standard SGD update
                        if p in self.masks:
                            mask = self.masks[p]
                            p.grad.mul_(mask) # In-place multiplication: grad = grad * mask
                    
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # Call the functional SGD step (standard PyTorch logic)
            # We must implement the manual update or call the functional API 
            # ensuring we use the modified gradients.
            
            # Since we modified p.grad in-place above, we can just call the standard 
            # SGD logic or implementing a simplified version here:
            
            for i, p in enumerate(params_with_grad):
                d_p = d_p_list[i]
                
                # Weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Momentum
                if momentum != 0:
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = d_p.clone().detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - group['dampening'] if 'dampening' in group else 1)
                    d_p = buf

                # Update step
                p.add_(d_p, alpha=-lr)

        return loss

# ---------------------------------------------------------
# Part 2: Fisher Information (Sensitivity) Calculation 
# ---------------------------------------------------------
def compute_fisher_sensitivity(model, dataloader, criterion, device, num_batches=10):
    """
    Computes the diagonal Fisher Information Matrix (FIM) scores.
    Sensitivity = Average of (Gradient ** 2)
    
    :param num_batches: Number of calibration rounds 
    :return: A dictionary {param: sensitivity_tensor}
    """
    model.eval() # Gradients are still computed in eval mode if we don't use no_grad
    sensitivity_scores = {}
    
    # Initialize accumulators
    for p in model.parameters():
        if p.requires_grad:
            sensitivity_scores[p] = torch.zeros_like(p.data)

    print(f"Calculating sensitivity over {num_batches} batches...")
    
    # Iterate over data
    processed_batches = 0
    for inputs, targets in dataloader:
        if processed_batches >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Accumulate squared gradients 
        # Diagonal Fisher Information ~ E[grad^2]
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                sensitivity_scores[p] += p.grad.data ** 2
        
        processed_batches += 1

    # Normalize by number of batches
    for p in sensitivity_scores:
        sensitivity_scores[p] /= processed_batches
        
    return sensitivity_scores

# ---------------------------------------------------------
# Part 3: Mask Calibration 
# ---------------------------------------------------------
def calibrate_masks(sensitivity_scores, sparsity_ratio=0.1, keep_least_sensitive=True):
    """
    Creates binary masks based on sensitivity scores.
    
    Task Arithmetic logic :
    We usually want to modify "low-sensitivity" parameters to avoid 
    interfering with pre-trained knowledge.
    
    :param sparsity_ratio: Percentage of parameters to UPDATE (Mask=1).
                           e.g., 0.1 means we update 10% of weights.
    :param keep_least_sensitive: 
           If True: Update the LOWEST sensitivity weights (Mask=1 where Sens is Low).
           If False: Update the HIGHEST sensitivity weights.
    :return: A dictionary {param: binary_mask_tensor}
    """
    masks = {}
    
    # We can compute the threshold globally or layer-wise. 
    # Global thresholding is common in Task Arithmetic literature.
    
    # 1. Flatten all scores to find the global threshold
    all_scores = torch.cat([s.view(-1) for s in sensitivity_scores.values()])
    
    # 2. Determine threshold
    # We want to select `sparsity_ratio` percent of parameters.
    num_params = all_scores.numel()
    k = int(num_params * sparsity_ratio)
    
    if keep_least_sensitive:
        # We want to update the k LEAST sensitive parameters.
        # So we look for the k-th smallest value.
        # weights < threshold -> Mask 1 (Update)
        # weights > threshold -> Mask 0 (Freeze)
        threshold = torch.kthvalue(all_scores, k).values.item()
        
        for p, score in sensitivity_scores.items():
            # Mask = 1 if score <= threshold (Low Sensitivity)
            # Mask = 0 if score > threshold
            mask = (score <= threshold).float()
            masks[p] = mask
    else:
        # (For Task 4 Extension) Update MOST sensitive
        # weights > threshold -> Mask 1
        threshold = torch.kthvalue(all_scores, num_params - k).values.item()
        
        for p, score in sensitivity_scores.items():
            mask = (score >= threshold).float()
            masks[p] = mask
            
    return masks
