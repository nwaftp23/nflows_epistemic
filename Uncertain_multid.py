import json
import os
import argparse
import sys
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import gym
import imageio

from replay_buffer import ReplayMemory, load_mem, load_mem_uncertain
from plot_functions import plot_input_distribution, plot_rmse_likelihood, plot_state_hm
from env import sim_env, load_noise
from analyze_fit import check_dyna_fit, calc_rmse, calc_kl_multid
from estimate_uncertainty import (check_uncertainty_wet_chicken, 
    check_uncertainty_pendulum, find_best_points)
from utils import (instantiate_model, normalize, un_normalize,
    identity, gen_folder_uncertain, seed_everything, natural_keys)
from policy import load_policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="WetChicken-v0",
                        help='Environment [WetChicken-v0, Pendulum-v0, HalfCheetah-v2, Hopper-v2]')
    parser.add_argument('--seed', type=int, default=1456,
                        help='random seed (default: 123456)')
    parser.add_argument('--noise_seed', type=int, default=14,
                        help='random seed (default: 123456)')
    parser.add_argument('--num_layers', default=3, help='total number of flows', type = int)
    parser.add_argument('--hids', type = int, default = 256, help='hidden units in flows')
    parser.add_argument('--lr', default=5e-4, type=float, help='flows learning rate')
    parser.add_argument('--gamma', default=0.999, type=float, help='schedule for lr step')
    parser.add_argument('--batch_size', default=2056, type=int, help='size of training batch size')
    parser.add_argument('--save_model', action= 'store_true', help='save model or not')
    parser.add_argument('--bins', type = int, default = 10, help='number of bins for spline NSF')
    parser.add_argument('--domain', type = float, default = 1.2, help='domain for spline NSF')
    parser.add_argument('--show', action= 'store_true', help='show graphs')
    parser.add_argument('--epochs', default=100, type=int, 
            help='number of epochs for dyna model')
    parser.add_argument('--dyna_model', default="nflows_ensemble", type =str,
            help='Selects the dynamics model [nflows, gp, mc_drop, nn_ensemble, nflows_ensemble])')
    parser.add_argument('--ensemble_size', default=5, type = int,
            help='number of components in uncertainty models')
    parser.add_argument('--bootstrap', action= 'store_true',
            help='whether or not to bootstrap the data, used for dyna_model PE')
    parser.add_argument('--noise_weight', type=float, default=0.0,
                        help='how much noise to add in')
    parser.add_argument('--modes', default=1, type=int,
            help='number of modes in noise to simulate chaotic dynamics')
    parser.add_argument('--valley_distribution', action= 'store_true',
            help='whether or not to create noise via the valley distribution')
    parser.add_argument('--fat_tail', action= 'store_true',
            help='whether or not to create noise via the fat tail distribution')
    parser.add_argument('--replay_size', type=int, default=1000000,
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--nstep', type=int, default=1,
                        help='the nth step to predict')
    parser.add_argument('--epochs_multiplier', type=int, default=100,
                        help='number of printouts')
    parser.add_argument('--policy_type', default='SAC', type=str,
                        help='pick type of police to run (SAC, LinearRand, PureRand)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(tau) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter alpha determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust alpha (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--data_size', type=int, default=200,
                        help='controls size of the data (negative number use all data)')
    parser.add_argument('--action_seq', action='store_true', 
                        help='sequence of actions to predict next state')
    parser.add_argument('--conditional_step', action="store_true",
                        help='condition on what step to predict')
    parser.add_argument('--rqs', action="store_true",
                        help='rational quadratic or cubic spline')
    parser.add_argument('--dropout_masks', action="store_true",
                        help='fixed set of dropout masks')
    parser.add_argument('--multihead', action="store_true",
                        help='multihead ensemble')
    parser.add_argument('--base_distro', action="store_true",
                        help='ensemble in base distro')
    parser.add_argument('--uncertain_nflows', action="store_true",
                        help='uncertainty in nflow layers')
    parser.add_argument('--rc_data', action="store_true",
                        help='rc car data or not')
    parser.add_argument('--index', type=int, default=-50,
                        help='Index for hyperparam list')
    parser.add_argument('--uncertainty_suffix', default="base",
                        help='[base, out, mean, max]')
    parser.add_argument('--test_acquisition', action="store_true", 
                        help='test different acquisitions')
    parser.add_argument('--acquisition_type', type=str, default='mutual_info_base',
                        help='how to acquire new points')
    args = parser.parse_args()
    if args.conditional_step:
        args.nstep = 1
    print(args)
    step_ahead_max = 30
    seed_everything(args.seed)
    store_dir = './results'
    save_model_dir = './models'
    output_preproc = normalize
    output_postproc = un_normalize
    input_preproc = normalize
    input_postproc = un_normalize
    branch_folder, child_folder = gen_folder_uncertain(args)
    env_dir = os.path.join('./replay_buffers', branch_folder)
    store_dir = os.path.join(env_dir, child_folder)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    save_model_dir = os.path.join(save_model_dir, branch_folder)
    save_model_dir = os.path.join(save_model_dir, child_folder)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    results_dir = os.path.join(store_dir, 'results/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    imgs_dir = os.path.join(store_dir, 'epoch_imgs/')
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    with open(os.path.join(store_dir, 'date_ran.txt'), mode='a') as f:
        f.write(f'Date: \n{time.strftime("%Y-%m-%d_%H_%M_%S")}')
    with open(os.path.join(save_model_dir, 'date_ran.txt'), mode='a') as f:
        f.write(f'Date: \n{time.strftime("%Y-%m-%d_%H_%M_%S")}')
    epoch_files = os.listdir(imgs_dir)
    for f in epoch_files:
        path = os.path.join(imgs_dir, f)
        os.remove(path)
    results_files = os.listdir(results_dir)
    for f in results_files:
        path = os.path.join(results_dir, f)
        os.remove(path)
    if args.dyna_model =='gp':
        device = 'cpu'
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env)
    memory = ReplayMemory(args.replay_size, args.batch_size, bootstrap = args.bootstrap,
            ensemble_size = args.ensemble_size, shuffle = True, action_seq = args.action_seq,
            conditional_step = args.conditional_step, step_ahead_max = step_ahead_max)
    buf_dir = load_mem_uncertain(args, memory, env_dir)
    test_memory = ReplayMemory(args.replay_size, 1028, bootstrap = args.bootstrap,
            ensemble_size = args.ensemble_size, shuffle = False, action_seq = args.action_seq, 
            conditional_step = args.conditional_step, step_ahead_max = step_ahead_max)
    buf_dir = load_mem_uncertain(args, test_memory, env_dir, test=True)
    memory.reduce_buffer(args.data_size)
    test_memory.reduce_buffer(2000)
    oracle_memory = ReplayMemory(args.replay_size, args.batch_size, 
        bootstrap = args.bootstrap, ensemble_size = args.ensemble_size, 
        shuffle = False, action_seq = args.action_seq, 
        conditional_step = args.conditional_step, step_ahead_max = step_ahead_max)
    buf_dir = load_mem_uncertain(args, oracle_memory, env_dir, oracle=True)
    oracle_memory.remove_portion(memory.buffer)
    if args.env in ['Pendulum-v0', 'WetChicken-v0']:
        plot_state_hm(memory, test_memory, imgs_dir, env=args.env, show=False)
    noise_params = load_noise(args, buf_dir)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if args.action_seq:
        context_dims = action_dim*args.nstep+state_dim
        action_dim_seq = action_dim*args.nstep
    else:
        context_dims = action_dim+state_dim
        action_dim_seq = action_dim
    if args.conditional_step:
        context_dims += 1
    if args.policy_type == 'SAC':
        load_model_dir = './pytorch-sac/models'
    else:
        load_model_dir = store_dir
    policy = load_policy(args, state_dim, action_dim, env, load_model_dir)
    args.output_dim = state_dim
    output_dim = state_dim
    args.context_dim = context_dims
    with open(os.path.join(store_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(save_model_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    dyna_model = instantiate_model(args, output_dim, context_dims, device, input_preproc,
        output_preproc, step_ahead_max)
    test_losses = []
    rmses = []
    kls = []
    numb_points_2_add = 10
    gp = False
    ensemble = False
    if args.dyna_model == 'nn_ensemble':
        numb_samps = 5000
        uncertainty_type = 'ensemble'
        ensemble = True
    elif args.dyna_model == 'mc_drop':
        numb_samps = 2500
        uncertainty_type = 'ensemble'
        ensemble = True
    elif args.dyna_model == 'nflows_ensemble':
        numb_samps = 5000
        uncertainty_type = 'nflows_ensemble'
        if not args.uncertain_nflows:
            uncertainty_type = 'nflows_ensemble'
        else:
            uncertainty_type = 'nflows_ensemble_out'
    elif args.dyna_model == 'nflows':
        numb_samps = 1000
        uncertainty_type = 'nflows'
    elif args.dyna_model == 'gp':
        numb_samps = 100
        uncertainty_type = 'gp'
        gp = True
    train_set_size = [len(memory.buffer)]
    for i in range(args.epochs_multiplier):
        start_time = time.time()
        train_loss = dyna_model.train(args.epochs, memory)
        dyna_model.detach_model()
        test_loss = dyna_model.loss(test_memory)
        epoch_suffix = 'epoch_'+str(((i+1)))
        step_ahead_max_pl = step_ahead_max
        if not args.conditional_step:
            step_ahead_max_pl = None
        samp_oracle = oracle_memory.sample(1000)
        samp_oracle = oracle_memory.sample(10)
        if args.acquisition_type != 'random':
            points_2_add = find_best_points(samp_oracle, uncertainty_type, numb_samps,
                dyna_model, input_preproc, args.ensemble_size,
                device, acquisition_criteria = args.acquisition_type)
        else:
            rand_samp = oracle_memory.sample(10)
            points_2_add = [(rand_samp[0][i], rand_samp[1][i], 
                rand_samp[2][i], rand_samp[3][i], rand_samp[4][i], 
                rand_samp[5][i], rand_samp[6][i]) for i in range(10)]
            torch.cuda.empty_cache()
        memory.add_to_buffer(points_2_add)
        oracle_memory.remove_portion(points_2_add)
        rmse = calc_rmse(test_memory, input_preproc, output_postproc,
            dyna_model, gp = gp, ensemble = ensemble,
            ensemble_size = args.ensemble_size, device = device)
        batch = [tpl for tpl in test_memory.buffer]
        states, actions, reward, next_states, done, noisy_actions, index = map(np.stack, zip(*batch))
        states = torch.tensor(states, dtype = torch.float32).to(device)
        actions = torch.tensor(actions, dtype = torch.float32).to(device)
        inps = torch.hstack([states, actions])
        subset = np.random.choice(inps.shape[0], size=50, replace=False)
        test_points = inps[subset, :]
        states = states[subset, :]
        actions = actions[subset, :]
        kl = calc_kl_multid(dyna_model, test_points, input_preproc, output_postproc, env,
            noise_params, policy, actions.cpu(), states.cpu(), args.nstep)
        test_losses += [test_loss]
        rmses.append(rmse)
        kls.append(np.array(kl)[(np.array(kl)!=np.inf)].mean())
        mean_dyna_loss = torch.tensor(train_loss).mean()
        test_likelihood = np.exp(-np.array(test_losses))
        plot_rmse_likelihood(train_loss, np.arange(len(train_loss)),
            'train_loss', store_dir=results_dir)
        plot_rmse_likelihood(test_losses, train_set_size,
            'test_loss', store_dir=results_dir)
        plot_rmse_likelihood(rmses, train_set_size, 'rmse', store_dir=results_dir)
        plot_rmse_likelihood(test_likelihood, train_set_size,
            'likelihood', store_dir=results_dir)
        plot_rmse_likelihood(kls, train_set_size, 'kl', store_dir=results_dir)
        train_set_size.append(len(memory.buffer))
        end_time = time.time()
        train_time = str(timedelta(seconds=(end_time-start_time)))
        performance_string = f'Total Epochs: {(i+1)}, '\
                             f'Train Loss: {mean_dyna_loss:.2f}, '\
                             f'test Loss: {test_loss:.2f}, '\
                             f'Train Time: {train_time}'
        print(performance_string)
        print(f'RMSE Test: {rmse}, KL Test:{np.array(kl)[(np.array(kl)!=np.inf)].mean():.2f},'\
            f' Train Set Size: {len(memory.buffer)-numb_points_2_add}')
        np.save(os.path.join(results_dir, ('train_loss_array')), np.array(train_loss))
        np.save(os.path.join(results_dir, ('test_loss_array')), np.array(test_losses))
        np.save(os.path.join(results_dir, ('rmse_array')), np.array(rmses))
        np.save(os.path.join(results_dir, ('kls_array')), np.array(kls))
        print('Saving Model')
        model_path = os.path.join(save_model_dir,('model.pt'))
        dyna_model.save_model(model_path)
        dyna_model = instantiate_model(args, output_dim, context_dims, device, input_preproc,
            output_preproc, step_ahead_max)
        print("-----------------------------------------------")
