import pickle
import time
from datetime import timedelta
import random
import os

import torch
import numpy as np
from scipy.special import expit
from scipy.stats import dirichlet
from plot_functions import plot_noise, plot_valley
from utils import seed_everything

MAX_STEPS = 1000

def sim_env(env, episodes, noise_params, rand_init=True, gym_state=None, 
            actions=[], replay_buffer=None, sample =False, 
            dyna_model = None, dyna_horizon = 10, policy = None, store =True, 
            deterministic_policy = True, open_loop=False):
    observe_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ground_truth = np.zeros([episodes, observe_dim, dyna_horizon])
    steps = 0
    #import pdb; pdb.set_trace()
    for j in range(episodes):
        done = False
        if open_loop:
            state_0, actions = sample_actions_seqs(env, policy)
            state = env.reset(state = state_0)
        elif rand_init:
            state = env.reset()
        else:
            state = env.reset(state = gym_state)
        i = 0
        tot_reward = 0
        #if j % 100 == 0:
        #    print(f'steps: {steps}')
        while not done:
            if (actions is not None) & (i<len(actions)):
                action = actions[i]
            elif policy:
                action = policy[0].select_action(state, replay_buffer, 
                    evaluate = deterministic_policy)
            else:
                action = env.action_space.sample()
            noise = get_noise(noise_params, env)
            noisy_action = action + noise
            unseen  = np.clip(noisy_action, -env.action_space.high, env.action_space.high)
            next_state, reward, done, _ = env.step(unseen)
            #else:
            #    print('noisy state does not work')
            #    next_state, reward, done, _ = env.step(action)
            #    unseen = next_state
            #    next_state = next_state + noise
            tot_reward += reward
            if store:
                replay_buffer.push(state, action, reward, next_state, done, unseen)
            if sample:
                ground_truth[j, :, i] = next_state
                if  (i+1) >= dyna_horizon:
                    done = True
            state = next_state
            i += 1
            if i >= MAX_STEPS:
                done = True
        steps += i
    return ground_truth, tot_reward, steps

def sample_actions_seqs(env, policy):
    done = False
    state = env.reset()
    state_0 = state
    actions = []
    while not done:
        action = policy[0].select_action(state, '',
                            evaluate = False)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        actions.append(action)
    return state_0, actions


def get_noise(noise_params, env):
    if noise_params:
        component_weights = noise_params[0]
        alphas = noise_params[1]
        betas = noise_params[2]
        distro = noise_params[3]
        noise_weight = noise_params[4]
        k_component = np.random.choice(len(component_weights), 1, p=component_weights)
        if distro == 'gauss':
            noise = (np.random.normal(alphas[0], betas[0], 1))*noise_weight
        elif distro == 'random':
            noise = (expit(np.random.normal(alphas[k_component], betas[k_component], 1))*2
                -1)*noise_weight
        else:
            noise = (np.random.beta(alphas[k_component], betas[k_component], 1)*2-1)*noise_weight
    else:
        noise = np.zeros(env.action_space.shape[0])
    return noise

def create_noise(args, store_dir, suffix, seed):
    seed_everything(seed)
    if args.valley_distribution:
        component_weights = np.array([0.5, 0.5])
        alphas = np.array([1,14])
        betas = np.array([14,1])
        plot_valley(component_weights, alphas, betas, show = args.show,
                store_dir = store_dir, suffix=suffix, noise_weight= args.noise_weight)
        noise_params = [component_weights, alphas, betas, 'valley']
    elif args.fat_tail:
        component_weights = np.array([1])
        alphas = np.array([2])
        betas = np.array([0.5])
        plot_valley(component_weights, alphas, betas, show = args.show,
                store_dir = store_dir, suffix=suffix, noise_weight= args.noise_weight)
        noise_params = [component_weights, alphas, betas, 'fat_tail']
    elif args.gauss_noise:
        component_weights = np.array([1.0])
        alphas = np.array([0.0]) # mean for gaussian 
        betas = np.array([1.0]) # std for gaussian
        noise_params = [component_weights, alphas, betas, 'gauss']
    else:
        if args.modes > 0:
            component_weights = dirichlet(np.ones(args.modes)).rvs()[0]
            alphas = np.random.normal(0,1,args.modes)
            betas = np.random.normal(0,0.5,args.modes)**2
            plot_noise(component_weights, alphas, betas, show = args.show,
                    store_dir = store_dir, suffix = suffix, noise_weight= args.noise_weight)
            noise_params = [component_weights, alphas, betas, 'random']
        else:
            noise_params = []
    if noise_params:
        noise_params.append(args.noise_weight)
    return noise_params

def load_noise(args, store_dir):
    filename = 'noiseparams_seed'+str(args.noise_seed)+'.pkl'
    filepath = os.path.join(store_dir, filename)
    file = open(filepath, 'rb')
    noise_params = pickle.load(file)
    file.close()
    return noise_params
