import time
import pdb
import argparse
import datetime

import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from sac_utils import *

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.4, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=3000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--bins', type=int, default=500, 
                    help='number of bins for nsf')
parser.add_argument('--num_flows', type=int, default=6, 
                    help='number of flows in the policy')
parser.add_argument('--cuda', action="store_false",
                    help='run on CUDA (default: True)')
parser.add_argument('--modes', default=0, type=int,
                    help='number of modes for noise')
parser.add_argument('--valley_distribution', action= 'store_true',
            help='whether or not to create noise via the valley distribution')
parser.add_argument('--fat_tail', action= 'store_true',
            help='whether or not to create noise via the fat tail distribution')
parser.add_argument('--nstep', default= 30, type=int, 
            help='whether or not to create noise via the fat tail distribution')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
best_score = -1e8
model_dir = './sac_models'
branch_folder = args.env+'_SAC'
if args.valley_distribution:
    branch_folder += '_valleydistro'
elif args.fat_tail:
    branch_folder += '_fattail'
else:
    branch_folder += '_modes'+str(args.modes)
model_dir  = os.path.join(model_dir, branch_folder)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

suffix = time.strftime("%Y-%m-%d_%H_%M_%S")+'_seed'+str(args.seed)
stored_name = args.env
env = gym.make(args.env)
seed_everything(args.seed)
print(args)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0
max_episode_length = 1000

rewards = []
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state, memory)  # Sample action from policy
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
                #print(action)
        noisy_action  = np.clip(action, -env.action_space.high, env.action_space.high)
        next_state, reward, done, _ = env.step(noisy_action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        if episode_steps >= max_episode_length:
            done=True
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            episode_steps = 0
            while not done:
                action = agent.select_action(state, memory, evaluate=True)
                noisy_action  = np.clip(action, -env.action_space.high, env.action_space.high)
                next_state, reward, done, _ = env.step(noisy_action)
                episode_reward += reward
                episode_steps += 1
                if episode_steps >= max_episode_length:
                    done = True
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes
        #rewards.append(avg_reward)
        #np.save(os.path.join(store_dir, ('reward_array_'+suffix)), np.array(rewards))
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        if avg_reward > best_score:
            best_score = avg_reward
            agent.save_model(stored_name, '.pt')

agent.save_model(stored_name, '.pt')
env.close()

