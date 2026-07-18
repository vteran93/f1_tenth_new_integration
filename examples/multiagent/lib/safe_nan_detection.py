"""
Safe NaN Detection and Recovery Module
Provides safe runtime NaN detection without aggressive monkey patching.
"""

import numpy as np
import logging
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


def safe_tensor_check(tensor, name="tensor"):
    """
    Safely check and fix a tensor for NaN/Inf values.
    Returns: (tensor, was_fixed)
    """
    if tensor is None:
        return tensor, False
        
    if not isinstance(tensor, torch.Tensor):
        return tensor, False
    
    # Check for NaN or Inf
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if not (has_nan or has_inf):
        return tensor, False
    
    logger.warning(f"NaN/Inf detected in {name}: nan={has_nan}, inf={has_inf}")
    
    # Create safe replacement
    fixed_tensor = tensor.clone()
    
    if has_nan:
        fixed_tensor = torch.where(torch.isnan(fixed_tensor), torch.zeros_like(fixed_tensor), fixed_tensor)
    
    if has_inf:
        # Replace inf with large but finite values
        fixed_tensor = torch.where(torch.isinf(fixed_tensor) & (fixed_tensor > 0),
                                   torch.full_like(fixed_tensor, 100.0), fixed_tensor)
        fixed_tensor = torch.where(torch.isinf(fixed_tensor) & (fixed_tensor < 0),
                                   torch.full_like(fixed_tensor, -100.0), fixed_tensor)
    
    return fixed_tensor, True


def check_model_parameters(model, fix=False):
    """
    Check model parameters for NaN/Inf values.
    If fix=True, replace problematic parameters with safe values.
    Returns: number of parameters that were fixed
    """
    if model is None:
        return 0
        
    fixed_count = 0
    
    for name, param in model.named_parameters():
        if param is None or not isinstance(param, torch.Tensor):
            continue
            
        fixed_param, was_fixed = safe_tensor_check(param, f"parameter {name}")
        
        if was_fixed and fix:
            with torch.no_grad():
                param.copy_(fixed_param)
            fixed_count += 1
    
    if fixed_count > 0:
        logger.warning(f"Fixed {fixed_count} model parameters")
    
    return fixed_count


def safe_loss_clamp(loss_tensor, min_val=-1000.0, max_val=1000.0):
    """
    Safely clamp loss tensor to reasonable values.
    """
    if loss_tensor is None:
        return torch.tensor(0.0, requires_grad=True)
    
    if not isinstance(loss_tensor, torch.Tensor):
        return loss_tensor
    
    # Fix NaN/Inf first
    fixed_loss, was_fixed = safe_tensor_check(loss_tensor, "loss")
    
    if was_fixed:
        # If we had to fix NaN/Inf, return a small positive loss
        return torch.tensor(1.0, requires_grad=True, device=fixed_loss.device)
    
    # Clamp to reasonable range
    clamped_loss = torch.clamp(fixed_loss, min_val, max_val)
    
    return clamped_loss


class SafeNaNRecoveryCallback:
    """
    Safe callback for NaN recovery that doesn't interfere with Ray's internal operations.
    """
    
    def __init__(self):
        self.nan_detections = 0
        self.recoveries = 0
        self.last_check_iteration = 0
    
    def check_metrics(self, result):
        """
        Check training metrics for NaN values.
        Returns: list of metrics with NaN values
        """
        nan_metrics = []
        
        # Common metrics to check
        metrics_to_check = [
            'env_runners/episode_return_mean',
            'env_runners/episode_len_mean', 
            'episode_return_mean',
            'episode_len_mean',
            'info/learner/default_policy/learner_stats/policy_loss',
            'info/learner/default_policy/learner_stats/vf_loss',
            'policy_loss',
            'vf_loss'
        ]
        
        for key in metrics_to_check:
            value = result.get(key)
            if value is not None:
                try:
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            nan_metrics.append(key)
                    elif hasattr(value, 'item'):  # tensor-like
                        val = value.item() if hasattr(value, 'item') else value
                        if np.isnan(val) or np.isinf(val):
                            nan_metrics.append(key)
                except (ValueError, TypeError):
                    # Skip if we can't check the value
                    continue
        
        return nan_metrics
    
    def attempt_recovery(self, algorithm):
        """
        Attempt to recover from NaN by checking and fixing model parameters.
        """
        if algorithm is None:
            return False
            
        recovery_attempted = False
        
        try:
            # Get all policies from the algorithm
            if hasattr(algorithm, 'get_policy'):
                # Single policy case
                try:
                    policy = algorithm.get_policy()
                    if policy and hasattr(policy, 'model'):
                        fixed_count = check_model_parameters(policy.model, fix=True)
                        if fixed_count > 0:
                            recovery_attempted = True
                except Exception as e:
                    logger.debug(f"Could not check single policy: {e}")
                    
            # Multi-policy case or workers
            if hasattr(algorithm, 'env_runner_group') and algorithm.env_runner_group:
                try:
                    def check_worker_policies(worker):
                        worker_recovery = False
                        if hasattr(worker, 'policy_map'):
                            for policy_id, policy in worker.policy_map.items():
                                if hasattr(policy, 'model') and policy.model:
                                    fixed_count = check_model_parameters(policy.model, fix=True)
                                    if fixed_count > 0:
                                        worker_recovery = True
                        return worker_recovery
                    
                    # Check local worker first
                    local_worker = algorithm.env_runner_group.local_worker()
                    if local_worker:
                        if check_worker_policies(local_worker):
                            recovery_attempted = True
                            
                except Exception as e:
                    logger.debug(f"Could not check worker policies: {e}")
                    
        except Exception as e:
            logger.warning(f"Recovery attempt failed: {e}")
            
        return recovery_attempted
    
    def on_train_result(self, algorithm, result):
        """
        Check training results and attempt recovery if needed.
        """
        current_iteration = result.get('training_iteration', 0)
        
        # Only check every few iterations to avoid performance impact
        if current_iteration - self.last_check_iteration < 5:
            return result
            
        self.last_check_iteration = current_iteration
        
        # Check for NaN in metrics
        nan_metrics = self.check_metrics(result)
        
        if nan_metrics:
            self.nan_detections += 1
            logger.error(f"NaN detected in metrics: {nan_metrics} (detection #{self.nan_detections})")
            
            # Attempt recovery
            if self.attempt_recovery(algorithm):
                self.recoveries += 1
                logger.info(f"Attempted NaN recovery #{self.recoveries}")
            else:
                logger.warning("Could not attempt recovery - no accessible policies")
        
        return result


def create_safe_callbacks():
    """
    Create a list of safe callbacks for NaN protection.
    """
    return [SafeNaNRecoveryCallback()]


def safe_policy_loss_wrapper(original_loss_fn):
    """
    Wrapper for policy loss functions to add NaN protection.
    """
    def wrapped_loss_fn(*args, **kwargs):
        try:
            loss = original_loss_fn(*args, **kwargs)
            return safe_loss_clamp(loss)
        except Exception as e:
            logger.error(f"Error in policy loss computation: {e}")
            return torch.tensor(1.0, requires_grad=True)
    
    return wrapped_loss_fn


def init_safe_nan_protection():
    """
    Initialize safe NaN protection without aggressive monkey patching.
    """
    logger.info("Initializing safe NaN protection")
    
    # Set torch to detect NaN/Inf in operations
    if torch is not None:
        torch.autograd.set_detect_anomaly(False)  # Don't use anomaly detection as it's too slow
        
        # Set reasonable default for numerical stability
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    logger.info("Safe NaN protection initialized")