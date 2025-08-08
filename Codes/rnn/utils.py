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