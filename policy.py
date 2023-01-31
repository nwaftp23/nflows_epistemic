import pdb
import os
import sys
sys.path.append('../pytorch-soft-actor-critic')
sys.path.append('../sac')

import torch
import torch.nn as nn

from sac import SAC

class LinearPolicy(nn.Module):
    ''' linear policy for simulating data
    '''
    def __init__(self, state_dim, action_dim):
        super(LinearPolicy, self).__init__()
        self.linear = torch.nn.Linear(state_dim, action_dim)

    def forward(self, state):
        a = self.linear(state)
        a = a.data.numpy().flatten()
        return a

    def select_action(self, state, memory, evaluate=True):
        state = torch.tensor(state).float()
        a = self.linear(state)
        a = a.data.numpy().flatten()
        a = a/4
        return a

def load_policy(args, state_dim, action_dim, env, store_dir):
    if args.policy_type == 'LinearRand':
        filename = args.policy_type+ '_seed' +str(args.noise_seed)+'.pt'
        filepath = os.path.join(store_dir, filename)
        policy = LinearPolicy(state_dim, action_dim)
        policy.load_state_dict(torch.load(filepath))
        policy = [policy]
    elif args.policy_type =='PureRand':
        policy = []
    else:
        actor = 'sac_actor_'+args.env+ '_' +'.pt'
        critic = 'sac_critic_'+args.env+ '_' +'.pt'
        policy_path = os.path.join(store_dir, actor)
        critic_path = os.path.join(store_dir, critic)
        agent = SAC(state_dim, env.action_space, args)
        agent.load_model(policy_path, critic_path)
        policy = [agent]
    return policy


