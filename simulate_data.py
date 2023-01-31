import pickle 
import os
import argparse
import sys
import time
sys.path.append('../pytorch-soft-actor-critic')

from tqdm import tqdm
import torch
import numpy as np
from scipy.stats import dirichlet
import gym

from env import sim_env, create_noise
from replay_buffer import ReplayMemory
from utils import seed_everything
from plot_functions import plot_noise, plot_valley
from policy import LinearPolicy, load_policy


def sim_policy(args, env, store_dir, suffix, memory, 
        noise_params, gym_state = [], policy =[]):
    state_dim = env.observation_space.shape[0]
    if args.policy_type == 'LinearRand':
        action_dim = env.action_space.shape[0]
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
        agent = LinearPolicy(**kwargs)
        policy = [agent]
    elif args.policy_type == 'PureRand':
        policy = []
    deterministic_policy = True
    if not args.test_data:
        deterministic_policy = False
    _, tot_reward, numb_steps = sim_env(env, args.numb_episodes, noise_params, 
            rand_init=True, replay_buffer= memory, policy = policy, 
            deterministic_policy = deterministic_policy)
    return policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(tau) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter alpha determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust alpha (default: False)')
    parser.add_argument('--seed', type=int, default=1456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--policy_type', default='SAC', type=str,
                        help='pick type of police to run (SAC, LinearRand, PureRand)')
    parser.add_argument("--numb_episodes", default=100, type=int,
                        help='number of episodes to collect data for')
    parser.add_argument('--env', default="Pendulum-v0",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--modes', default=1, type=int,
            help='number of modes in noise to simulate chaotic dynamics')
    parser.add_argument('--valley_distribution', action= 'store_true',
            help='whether or not to create noise via the valley distribution')
    parser.add_argument('--fat_tail', action= 'store_true',
            help='whether or not to create noise via the fat tail distribution')
    parser.add_argument('--gauss_noise', action= 'store_true',
            help='whether or not to create noise with 0 mean gaussian')
    parser.add_argument('--ensemble_size', default=5, type = int,
            help='ensemble size for pets')
    parser.add_argument('--bootstrap', action= 'store_true',
            help='whether or not to bootstrap the data')
    parser.add_argument('--show', action= 'store_true',
            help='show graphs or save them')
    parser.add_argument('--noise_weight', type=float, default=0.2, 
                        help='how much noise to add in')
    parser.add_argument('--test_data', action= 'store_true',
            help='create test data set')
    parser.add_argument('--noise_seed', type=int, default=14, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--noisy_state', action = 'store_true',
                        help='noise on the state or the action')
    args = parser.parse_args()
    print(args)
    store_dir = './replay_buffers'
    branch_folder = args.env
    branch_folder = os.path.join(branch_folder, 'noiseweight'+str(args.noise_weight)+'_')
    if args.fat_tail:
        branch_folder = branch_folder+'fattail'
    elif args.valley_distribution:
        branch_folder = branch_folder+'valleydistribution'
    else:
        branch_folder = branch_folder+'modes'+str(args.modes)
    store_dir = os.path.join(store_dir, branch_folder)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    suffix = 'seed'+str(args.noise_seed)

    env = gym.make(args.env)

    policy=[]
    if args.policy_type == 'SAC':
        model_dir ='./pytorch-sac/models'
        state_dim = env.observation_space.shape[0]
        action_dim = 42 
        policy = load_policy(args, state_dim, action_dim, env, model_dir)
    noise_params = create_noise(args, store_dir, suffix, args.noise_seed)    
    
    memory = ReplayMemory(args.replay_size, args.batch_size, bootstrap = args.bootstrap,
            ensemble_size = args.ensemble_size, shuffle = False)
    if args.test_data:
        seed_everything(args.seed+42)
    else:
        seed_everything(args.seed)
    policy = sim_policy(args, env, store_dir, suffix, memory, noise_params,  policy=policy)
    if args.test_data:
        buffer_path = os.path.join(store_dir, (args.policy_type + '_buffertest_'+suffix+ '.pkl'))
    else:
        buffer_path = os.path.join(store_dir, (args.policy_type + '_buffer_'+suffix+ '.pkl'))
    with open(buffer_path, 'wb') as f:
        pickle.dump(memory.buffer, f)
    f.close()
    noise_path = os.path.join(store_dir, ('noiseparams_'+suffix+ '.pkl'))
    with open(noise_path, 'wb') as noise:
        pickle.dump(noise_params, noise)
    f.close()
    if args.policy_type =='LinearRand':
        policy_path = os.path.join(store_dir, (args.policy_type +'_'+suffix+'.pt')) 
        torch.save(policy[0].state_dict(), policy_path)
