import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_mc_drop(num_layers=2, hids=20, dims=2, context_dims=2,
        device = 'cuda'):
    model = Linear_MC_Drop(context_dims, dims, hids, 0.5, 0,
         mc_drop = True, device = device, num_layers = num_layers)

    return model

def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)

class Linear_MC_Drop(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid, pdrop, context_dim, 
            mc_drop = True, device='cpu', num_layers = 2):
        super(Linear_MC_Drop, self).__init__()

        self.pdrop = pdrop

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hid = n_hid
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim+context_dim, n_hid)
        self.fc = []
        for i in range(num_layers-1):
            exec(f'self.fc{(i+2)} = nn.Linear(n_hid, n_hid)')
            exec(f'self.fc.append(self.fc{(i+2)})')
        self.mean_linear = nn.Linear(n_hid, output_dim)
        self.log_std_linear = nn.Linear(n_hid, output_dim)

        self.act = nn.Tanh()
        #self.act = nn.SELU()
        self.mc_drop = mc_drop

    def forward(self, x, context=None, rand_mask=True, mask_index = 0, debug=False):
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        if context is None:
            pass
        else:
            x = torch.cat((x, context), dim=1)
        # -----------------
        if debug:
            import pdb; pdb.set_trace()
        x = self.fc1(x)
        if self.mc_drop:
            x = MC_dropout(x, p=self.pdrop, mask=rand_mask)
        mask_prob=self.pdrop**(x==0).sum(1)
        mask_prob*=((1-self.pdrop)**(x!=0).sum(1))
        # -----------------
        x = self.act(x)
        # -----------------
        for lay in self.fc:
            x = lay(x) 
            if self.mc_drop:
                x = MC_dropout(x, p=self.pdrop, mask=rand_mask)
            mask_prob*=(self.pdrop**(x==0).sum(1))
            mask_prob*=((1-self.pdrop)**(x!=0).sum(1))
            # -----------------
            x = self.act(x)
            # -----------------
        mu = self.mean_linear(x)
        sig = torch.exp(self.log_std_linear(x))
        return mu, sig, mask_prob

    def sample(self, numb_samps, context=None, kwargs={}):
        mu, sig, mask_prob = self.forward(context)
        norm_rv = torch.distributions.normal.Normal(mu, sig)
        samp = norm_rv.sample([numb_samps])
        # shoulb be (numb_conditionals, numb_samps, numb_dimensions)
        samp = samp.reshape(samp.shape[1], samp.shape[0], samp.shape[2])
        return samp, mask_prob, mu, sig 

    def sample_and_log_prob(self, numb_samps, context=None, kwargs={}):
        mu, sig, mask_prob = self.forward(context)
        norm_rv = torch.distributions.normal.Normal(mu, sig)
        samp = norm_rv.sample([numb_samps])
        log_prob = norm_rv.log_prob(samp).sum(2)
        return samp, log_prob, mask_prob, mu, sig

    def loss_val(self, mu, sig, outs, reduction='mean'):
        criterion = torch.nn.GaussianNLLLoss(reduction=reduction, full=True)
        loss = -criterion(outs, mu, sig**2)
        return loss

    def log_prob(self, mu, sig, outs):
        norm_rv = torch.distributions.normal.Normal(mu, sig)
        if outs.shape[1]>1:
            log_prob = norm_rv.log_prob(outs).sum(1)
        else:
            log_prob = norm_rv.log_prob(outs)
        return log_prob
