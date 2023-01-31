import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import nflows
from nflows import distributions, transforms, utils, flows
from nflows.nn import nets



LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def build_model(num_layers=2, hids=100, dims=2, context_dims=2,
        batch_norm=False, activation=torch.nn.functional.relu, bins = 5, tail=3.0):
    # initialize normalizing flow model
    context_net = nn.Sequential(nn.Linear(context_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*dims)
        )
    base_dist = nflows.distributions.ConditionalDiagonalNormal(
        shape=[dims], context_encoder= context_net)

    transforms = []

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_dims,
            hidden_features=hids, num_blocks=2,
            use_batch_norm=batch_norm,
            activation=activation)

    for _ in range(num_layers):
        transforms.append(nflows.transforms.ReversePermutation(features=dims))
        # affine flows
#         transforms.append(nflows.transforms.MaskedAffineAutoregressiveTransform(
#             features=dims, 
#             hidden_features=hids,
#             context_features=context_dims,
#             use_batch_norm=batch_norm,
#             activation=activation
#         ))
        # coupling nsf
        if dims > 1:
            mask = nflows.utils.torchutils.create_mid_split_binary_mask(dims)
            transforms.append(
                nflows.transforms.PiecewiseRationalQuadraticCouplingTransform(
                    mask, create_net, tails='linear', num_bins=bins, tail_bound=tail,
                ))
        # masked autoregressive nsf
        if dims == 1:
           transforms.append(
                nflows.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                 features=dims,
                 hidden_features=hids,
                 context_features=context_dims,
                 tails='linear',
                 use_batch_norm=batch_norm,
                 num_bins=bins,
                 tail_bound = tail,
                 activation = activation))
    transforms.append(nflows.transforms.InverseTransform(nflows.transforms.Tanh()))
    transform = nflows.transforms.CompositeTransform(transforms)

    flow = nflows.flows.Flow(transform, base_dist)
    return flow



class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

class FlowPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, bins, action_space, num_flows):
        super(FlowPolicy, self).__init__()
        
        self.max_action = action_space.high.max()
        self.tail = int(np.ceil(np.max([action_space.high,np.abs(action_space.low)])))
        self.flow = build_model(num_layers=num_flows, dims=num_actions, hids=hidden_dim, 
                context_dims=num_inputs, bins=bins, tail=self.tail)
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def sample(self, state):
        #best_act = self.flow._distribution.mean(context=state)
        #best_act, _ = self.flow._transform.forward(best_act, context=state)
        action, log_prob = self.flow.sample_and_log_prob(1, context=state)
        action = action.squeeze(1)*self.max_action
        log_prob = log_prob.reshape(-1,1)/self.max_action
        return action, log_prob, action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(FlowPolicy, self).to(device)
