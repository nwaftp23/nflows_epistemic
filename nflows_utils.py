import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.stats import entropy, gaussian_kde, normaltest
import nflows
from nflows import distributions, transforms, utils, flows
from nflows.transforms.normalization import BatchNorm
from nflows.nn import nets
from nflows.transforms.base import (
    CompositeTransform,
    InputOutsideDomain,
    InverseTransform,
    Transform,
)
from nflows.utils import torchutils

def build_nflows(num_layers=2, hids=20, dims=2, context_dims=2, 
        batch_norm=False, activation=torch.nn.functional.relu, bins = 15, tail=8.0,  
        device = 'cuda', rqs=True, bimodal=False):
    context_net = Linear_2L(context_dims, 2*dims, hids, 0.5, 0,
         mc_drop = False, fixed_masks = False, 
         different_heads = False, device = device)
    base_dist = nflows.distributions.ConditionalDiagonalNormal(
        shape=[dims], context_encoder= context_net)

    transforms = []
    
    def create_net(in_features, out_features):
        return Linear_2L(in_features, out_features, hids, 0.5,
                context_dims, fixed_masks = False, 
                different_heads = False, device=device)
    for _ in range(num_layers):
        if dims > 1:
            transforms.append(nflows.transforms.RandomPermutation(features=dims))
            mask = nflows.utils.torchutils.create_mid_split_binary_mask(dims)
            transforms.append(
                    nflows.transforms.PiecewiseCubicCouplingTransform(mask, create_net, 
                        tails='linear', num_bins=bins, tail_bound=tail,
                ))
        if dims == 1:
            transforms.append(
                nflows.transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
                 features=dims,
                 hidden_features=hids,
                 context_features=context_dims,
                 num_blocks = 2,
                 use_batch_norm=batch_norm,
                 num_bins=bins,
                 tails='linear',
                 tail_bound = tail,
                 activation = activation,
                 use_residual_blocks = False,)) 
    transform = nflows.transforms.CompositeTransform(transforms)

    flow = nflows.flows.Flow(transform, base_dist)
    return flow

def build_nflows_ensemble(num_layers=2, hids=20, dims=2, context_dims=2, 
        batch_norm=False, activation=torch.nn.functional.relu, bins = 15, tail=8.0,  
        device = 'cuda', rqs=True, base = True, flows = True, multihead=False, 
        fixed_masks=False, ensemble_size=15, bimodal=False):
    if base:
        context_net = Linear_2L(context_dims, 2*dims, hids*2, 0.5, 0,
            fixed_masks = fixed_masks, num_masks = ensemble_size,
            different_heads = multihead, device = device)
    else:
        context_net = Linear_2L(context_dims, 2*dims, hids*2, 0.5, 0,
             fixed_masks = False, num_masks = ensemble_size,
             different_heads = False, device = device)
    base_dist = nflows.distributions.ConditionalDiagonalNormal(
        shape=[dims], context_encoder= context_net)

    transforms = []

    
    if flows:
        def create_net(in_features, out_features):
            return Linear_2L(in_features, out_features, hids, 0.5,
                    context_dims, fixed_masks=fixed_masks, 
                    different_heads = multihead, num_masks=ensemble_size, device=device)
    else:
        def create_net(in_features, out_features):
            return Linear_2L(in_features, out_features, hids, 0.5,
                    context_dims, fixed_masks = False, 
                    different_heads = False, device=device)
    for _ in range(num_layers):
        if dims > 1:
            transforms.append(nflows.transforms.RandomPermutation(features=dims))
            mask = nflows.utils.torchutils.create_mid_split_binary_mask(dims)
            transforms.append(
                    nflows.transforms.PiecewiseCubicCouplingTransform(mask, create_net, 
                        tails='linear', num_bins=bins, tail_bound=tail,
                ))
        if dims == 1:
            transforms.append(
                nflows.transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
                 features=dims,
                 hidden_features=hids,
                 context_features=context_dims,
                 num_blocks = 1,
                 use_batch_norm=batch_norm,
                 num_bins=bins,
                 tails='linear',
                 tail_bound = tail,
                 activation = activation,
                 use_residual_blocks = False,
                 ensemble = flows)) 
                 #create_context_net = create_net))
    transform = nflows.transforms.CompositeTransform(transforms)

    flow = nflows.flows.Flow(transform, base_dist)
    return flow


class Linear_2L(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, pdrop, context_dim, 
                 fixed_masks = False, num_masks = 10, different_heads = False, 
                 device='cpu'):
        super(Linear_2L, self).__init__()

        self.pdrop = pdrop

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hid = n_hid

        self.fc1 = nn.Linear(input_dim+context_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        if different_heads:
            self.heads = []
            for i in range(num_masks):
                exec(f'self.head{i} = nn.Linear(n_hid, output_dim)')
                exec(f'self.heads.append(self.head{i})')
        else:
            self.fc3 = nn.Linear(n_hid, output_dim)

        self.different_heads = different_heads
        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)
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
            y = self.heads[head_idx](x)
        else:
            y = self.fc3(x)

        return y
    
    def create_masks(self, num_masks, device):
        masks = []
        for i in range(num_masks):
            mask_l1 = torch.bernoulli(torch.full_like(torch.ones(self.n_hid), self.pdrop))\
                .to(device)
            mask_l2 = torch.bernoulli(torch.full_like(torch.ones(self.n_hid), self.pdrop))\
                .to(device)
            masks.append([mask_l1, mask_l2])
        self.masks = masks
