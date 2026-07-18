"""
NaN Detection and Recovery Module
Provides runtime NaN detection and correction for PPO training stability.
"""

import numpy as np
import logging
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


class NaNProtectedTorchDiagGaussian(TorchDiagGaussian):
    """
    Protected version of TorchDiagGaussian that handles NaN values.
    """
    
    def __init__(self, inputs, model=None):
        # Check for NaN in inputs before creating distribution
        mean, log_std = torch.chunk(inputs, 2, dim=-1)
        
        # Detect NaN values
        has_nan_mean = torch.isnan(mean).any()
        has_nan_std = torch.isnan(log_std).any()
        
        if has_nan_mean or has_nan_std:
            logger.warning(f"NaN detected in action distribution inputs! mean_nan: {has_nan_mean}, std_nan: {has_nan_std}")
            
            # Replace NaN values with safe defaults
            if has_nan_mean:
                mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
                logger.warning("Replaced NaN mean values with zeros")
            
            if has_nan_std:
                # Replace NaN log_std with small negative values (std ≈ 0.01)
                log_std = torch.where(torch.isnan(log_std), torch.full_like(log_std, -4.6), log_std)
                logger.warning("Replaced NaN log_std values with -4.6 (std ≈ 0.01)")
            
            # Clamp extreme values
            mean = torch.clamp(mean, -10.0, 10.0)
            log_std = torch.clamp(log_std, -20.0, 2.0)
            
            # Reconstruct inputs
            inputs = torch.cat([mean, log_std], dim=-1)
        
        # Call parent constructor with cleaned inputs
        super().__init__(inputs, model)


def add_nan_hooks(model):
    """
    Add forward hooks to detect NaN values in model parameters and outputs.
    """
    def nan_detection_hook(module, input, output):
        """Hook to detect NaN in forward pass."""
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                logger.error(f"NaN detected in {module.__class__.__name__} output!")
                # Replace NaN with zeros
                output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
                return output
        elif isinstance(output, (list, tuple)):
            cleaned_output = []
            for i, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                    logger.error(f"NaN detected in {module.__class__.__name__} output[{i}]!")
                    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
                cleaned_output.append(tensor)
            return type(output)(cleaned_output)
        return output
    
    # Register hooks on all modules
    for name, module in model.named_modules():
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            module.register_forward_hook(nan_detection_hook)
    
    return model


def check_and_fix_model_parameters(model):
    """
    Check model parameters for NaN/Inf and replace with safe values.
    """
    fixed_params = 0
    
    for name, param in model.named_parameters():
        if param is None:
            continue
            
        # Check for NaN or Inf
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"NaN/Inf detected in parameter: {name}")
            
            # Replace with Xavier uniform initialization
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.normal_(param, 0.0, 0.01)
            
            fixed_params += 1
            logger.warning(f"Reset parameter {name} with safe initialization")
    
    if fixed_params > 0:
        logger.warning(f"Fixed {fixed_params} parameters with NaN/Inf values")
    
    return fixed_params > 0


def safe_loss_computation(loss_tensor):
    """
    Safely compute loss, handling NaN and extreme values.
    """
    if loss_tensor is None:
        return torch.tensor(0.0, requires_grad=True)
    
    # Check for NaN
    if torch.isnan(loss_tensor).any():
        logger.error("NaN detected in loss computation!")
        return torch.tensor(1.0, requires_grad=True)  # Return a small positive loss
    
    # Check for extreme values
    if torch.isinf(loss_tensor).any():
        logger.error("Inf detected in loss computation!")
        return torch.tensor(1.0, requires_grad=True)
    
    # Clamp loss to reasonable range
    clamped_loss = torch.clamp(loss_tensor, -1000.0, 1000.0)
    
    if not torch.allclose(loss_tensor, clamped_loss):
        logger.warning("Clamped extreme loss values")
    
    return clamped_loss


def patch_action_distribution():
    """
    Monkey patch the action distribution to use NaN-protected version.
    """
    import ray.rllib.models.torch.torch_action_dist as torch_action_dist
    
    # Replace the original class
    torch_action_dist.TorchDiagGaussian = NaNProtectedTorchDiagGaussian
    logger.info("Patched TorchDiagGaussian with NaN protection")


class NaNRecoveryCallback:
    """
    Callback to handle NaN recovery during training.
    """
    
    def __init__(self):
        self.nan_count = 0
        self.recovery_count = 0
    
    def on_train_result(self, algorithm, result, **kwargs):
        """Check training results for NaN values."""
        
        # Check common metrics for NaN
        nan_metrics = []
        
        for key in ['episode_return_mean', 'episode_len_mean', 'policy_loss', 'vf_loss']:
            if key in result and result[key] is not None:
                if isinstance(result[key], (int, float)):
                    if np.isnan(result[key]) or np.isinf(result[key]):
                        nan_metrics.append(key)
                elif hasattr(result[key], 'isnan'):
                    if result[key].isnan() or result[key].isinf():
                        nan_metrics.append(key)
        
        if nan_metrics:
            self.nan_count += 1
            logger.error(f"NaN detected in metrics: {nan_metrics}")
            
            # Try to recover by resetting model parameters
            for policy in algorithm.get_policy().values():
                if hasattr(policy, 'model'):
                    if check_and_fix_model_parameters(policy.model):
                        self.recovery_count += 1
                        logger.info(f"Attempted parameter recovery #{self.recovery_count}")
        
        return result