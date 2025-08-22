import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.autograd import Function
import torch.nn.functional as F

class CategoricalMasked(Categorical):
    """
    A torch Categorical class with action masking.
    """

    def __init__(self, logits, mask):
        self.mask = mask

        if mask is None:
            super(CategoricalMasked, self).__init__(logits = logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype = logits.dtype
            )
            logits = torch.where(self.mask, logits, self.mask_value)
            super(CategoricalMasked, self).__init__(logits = logits)


    def entropy(self):
        if self.mask is None:
            return super().entropy()
        
        p_log_p = self.logits * self.probs

        # compute entropy with possible actions only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype = p_log_p.dtype, device = p_log_p.device),
        )

        return -torch.sum(p_log_p, axis = 1)
    

class FlattenExtractor(nn.Module):
    """
    A flatten feature extractor.
    """
    def forward(self, x):
        # keep the first dimension while flatten other dimensions
        return x.view(x.size(0), -1)


class ValueNet(nn.Module):
    """
    Value baseline network.
    """
    
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.fc_value = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        value = self.fc_value(x) # (batch_size, 1)

        return value


class ActionNet(nn.Module):
    """
    Action network.
    """

    def __init__(self, input_dim, output_dim):
        super(ActionNet, self).__init__()
        self.fc_action = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, mask = None):
        self.logits = self.fc_action(x) # record logits for later analyses

        # no action masking
        if mask == None:
            dist = Categorical(logits = self.logits)
        
        # with action masking
        elif mask != None:
            dist = CategoricalMasked(logits = self.logits, mask = mask)
        
        policy = dist.probs # (batch_size, output_dim)
        action = dist.sample() # (batch_size,)
        log_prob = dist.log_prob(action) # (batch_size,)
        entropy = dist.entropy() # (batch_size,)
        
        return action, policy, log_prob, entropy

def three_phase_linear(total, start, final, mid_ratio=0.4, dtype=np.float32):
    """
    Three-phase linear schedule.
    Args:
        total: total number of steps
        start: start value
        final: final value
        mid_ratio: ratio of middle phase
        dtype: data type
    """
    assert 0.0 < mid_ratio < 1.0
    first = total // 2
    mid   = int(round(total * mid_ratio))
    last  = total - first - mid
    if last < 0:  # 极端参数保护
        mid = total - first
        last = 0

    first_seg = np.full(first, start, dtype=dtype)
    if mid > 0:
        mid_seg = np.linspace(start, final, mid, dtype=dtype, endpoint=True)
    else:
        mid_seg = np.empty(0, dtype=dtype)
    last_seg  = np.full(last, final, dtype=dtype)

    sched = np.concatenate([first_seg, mid_seg, last_seg], axis=0)
    assert len(sched) == total
    return sched

def two_phase_linear(total: int, start: float, final: float, frac: float = 0.5, dtype=np.float32) -> np.ndarray:
    """
    Build a schedule of length `total`:
      - First `frac` steps: linearly decay from `start` to `final`
      - Last `1 - frac` steps: hold constant at `final`

    Args
    ----
    total : int
        Total number of steps (== number of batches in your training loop).
    start : float
        Starting value (e.g., initial learning rate / entropy coefficient).
    final : float
        Final value to reach at halfway point and hold afterward.
    dtype : numpy dtype
        Output dtype, default float32.

    Returns
    -------
    sched : np.ndarray of shape (total,)
    """
    if total <= 0:
        return np.empty(0, dtype=dtype)

    first = int(total * frac)
    rest = total - first

    first_seg = np.linspace(start, final, num=first, dtype=dtype, endpoint=False) if first > 0 \
                 else np.empty(0, dtype=dtype)
    last_seg = np.full(rest, final, dtype=dtype)

    sched = np.concatenate([first_seg, last_seg], axis=0)
    assert len(sched) == total
    return sched