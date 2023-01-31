import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def build_nn_ensemble(hids, dims, context_dims, device = 'cuda', 
        multihead=False, fixed_masks=False, ensemble_size=15):
    model = Linear_2L(context_dims, dims, hids*2, 0.5, 0,
        fixed_masks = fixed_masks, num_masks = ensemble_size,
        different_heads = multihead, device = device)
    return model 


class Linear_2L(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, pdrop, context_dim, 
                fixed_masks = False, num_masks = 10,
                different_heads = False, device='cpu', base=False):
        super(Linear_2L, self).__init__()

        self.pdrop = pdrop

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hid = n_hid

        self.fc1 = nn.Linear(input_dim+context_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        if different_heads:
            self.mean_heads = []
            self.log_std_heads = []
            for i in range(num_masks):
                exec(f'self.mean_linear{i} = nn.Linear(n_hid, output_dim)')
                exec(f'self.mean_heads.append(self.mean_linear{i})')
                exec(f'self.log_std_linear{i} = nn.Linear(n_hid, output_dim)')
                exec(f'self.log_std_heads.append(self.log_std_linear{i})')
        else:
            self.mean_linear = nn.Linear(n_hid, output_dim)
            self.log_std_linear = nn.Linear(n_hid, output_dim)
        self.different_heads = different_heads
        self.act = nn.ReLU(inplace=True)
        self.fixed_masks = fixed_masks
        if fixed_masks:
            self.create_masks(num_masks, device)
        self.num_masks = num_masks

    def forward(self, x, context=None, rand_mask=True, mask_index = 0):
        if self.fixed_masks:
            if rand_mask:
                mask = self.masks[np.random.choice(self.num_masks)]
            else:
                mask = self.masks[mask_index]
                
        if self.different_heads:
            if rand_mask:
                head_idx = np.random.choice(self.num_masks)
            else:
                head_idx = mask_index
        
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        if context is None:
            pass
        else:
            x = torch.cat((x, context), dim=1)
        # -----------------
        x = self.fc1(x)
        if self.fixed_masks:
             x = mask[0].repeat(x.shape[0],1)*x
        # -----------------
        x = self.act(x)
        # -----------------
        x = self.fc2(x)
        if self.fixed_masks:
            x = mask[1].repeat(x.shape[0],1)*x
        # -----------------
        x = self.act(x)
        # -----------------
        if self.different_heads:
            mu = self.mean_heads[head_idx](x)
            log_sig = self.log_std_heads[head_idx](x)
        else:
            mu = self.mean_linear(x)
            log_sig = self.log_std_linear(x)
        log_sig = torch.clamp(log_sig, min=LOG_SIG_MIN, max=LOG_SIG_MAX) 
        sig = torch.exp(log_sig)
        return mu, sig, 1/self.num_masks
    
    def create_masks(self, num_masks, device):
        masks = []
        for i in range(num_masks):
            mask_l1 = torch.bernoulli(torch.full_like(torch.ones(self.n_hid), self.pdrop))\
                .to(device)
            mask_l2 = torch.bernoulli(torch.full_like(torch.ones(self.n_hid), self.pdrop))\
                .to(device)
            masks.append([mask_l1, mask_l2])
        self.masks = masks

    def sample(self, numb_samps, context=None, kwargs={}):
        mu, sig, mask_prob = self.forward(context)
        norm_rv = torch.distributions.normal.Normal(mu, sig)
        samp = norm_rv.sample([numb_samps])
        samp = samp.reshape(samp.shape[1], samp.shape[0], samp.shape[2])
        return samp, mask_prob, mu, sig

    def sample_and_log_prob(self, numb_samps, context=None, kwargs={}):
        mu, sig, mask_prob = self.forward(context, **kwargs)
        norm_rv = torch.distributions.normal.Normal(mu, sig)
        samp = norm_rv.sample([numb_samps])
        log_prob = norm_rv.log_prob(samp).sum(2)
        return samp, log_prob, mask_prob, mu, sig

    def loss_val(self, mu, sig, outs, reduction='mean', full=False):
        criterion = torch.nn.GaussianNLLLoss(reduction=reduction, full=full)
        loss = -criterion(outs, mu, sig**2)
        return loss    

    def log_prob(self, mu, sig, outs):
        norm_rv = torch.distributions.normal.Normal(mu, sig)
        if outs.shape[1]>1:
            log_prob = norm_rv.log_prob(outs).sum(1)
        else:
            log_prob = norm_rv.log_prob(outs)
        return log_prob
