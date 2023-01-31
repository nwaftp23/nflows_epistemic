import gc
from copy import copy
import random
import os
import sys
import itertools
from time import time
from datetime import timedelta

import torch
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy 
from sklearn.neighbors import KernelDensity

from plot_functions import (plot_states, plot_state1, plot_noise)
from env import sim_env
from kl_estimator import naive_estimator
from envs_1d import hetero_samp_condition, bimodal_samp_condition

def get_thetas(states):
   thetas = np.arctan2(states[:,1], states[:,0])
   return thetas

def sample_gt_model(env, noise_params, model, replay_buffer, device, show, 
        store_dir, observe_dim, action_dim, action_dim_seq, input_preproc, output_postproc, 
        sample='random', inp_stats=None, out_stats=None, policy=[], 
        dyna_horizon=10, state = [], action = [], 
        step_ahead_max=False):
    if sample == 'replay':
        state, action = sample_trajectory_replay_buffer(replay_buffer, 1, observe_dim,
                                                        action_dim_seq)
        check_action = action.flatten().numpy()
        check_state = state
    elif sample == 'pair':
        check_state = state
        check_action = action
    else:
        if env.unwrapped.spec.id == 'Pendulum-v0':
            high = np.array([np.pi, 1])
            check_state = np.random.uniform(low=-high, high=high)
            check_action = np.random.uniform(-env.action_space.high, env.action_space.high)
            test_state = np.array([[np.cos(check_state[0]),np.sin(check_state[0]), 
                            check_action[0]]])
        elif env.unwrapped.spec.id == 'HalfCheetah-v2':
            qpos = np.random.uniform(low=-.1, high=.1, size=9)
            qvel = np.random.randn(9) *.1
            check_state = np.concatenate([qpos,qvel])
            check_action = np.random.uniform(-env.action_space.high, env.action_space.high)
        elif env.unwrapped.spec.id == 'Hopper-v2':
            qpos = (np.array([0.  , 1.25, 0.  , 0.  , 0.  , 0.  ]) + 
                np.random.uniform(low=-.005, high=.005, size=6))
            qvel = np.random.uniform(low=-.005, high=.005, size=6)
            check_state = np.concatenate([qpos,qvel])
            check_action = np.random.uniform(-env.action_space.high, env.action_space.high)
        elif env.unwrapped.spec.id == 'Swimmer-v2':
            qpos = np.random.uniform(low=-.1, high=.1, size=5)
            qvel = np.random.uniform(low=-.1, high=.1, size=5)
            check_state = np.concatenate([qpos,qvel])
            check_action = np.random.uniform(-env.action_space.high, env.action_space.high)
    if action_dim != action_dim_seq:
        actions = check_action.reshape(dyna_horizon, action_dim)
    else:
        actions = check_action.reshape(1, action_dim)
    if step_ahead_max:
        dyna_horizon = np.random.randint(step_ahead_max)+1
    data,_,_ = sim_env(env, 1000, noise_params, rand_init=False, policy=policy, 
        gym_state = check_state, actions = actions, sample=True, 
        dyna_horizon = dyna_horizon, store=False)
    data = data[:,:,-1]
    context_state = env.reset(state=check_state)
    if check_action.shape ==():
        check_action = np.expand_dims(check_action,axis=0)
    context = torch.tensor(np.concatenate([context_state, check_action]))\
            .type(torch.float32).to(device)
    context = input_preproc(context, inp_stats)
    if step_ahead_max:
        step_ahead_preproc = dyna_horizon/replay_buffer.episode_length
        step_ahead_preproc = torch.tensor([step_ahead_preproc], dtype=torch.float32)\
                        .to(device)
        context = torch.hstack([context, step_ahead_preproc])
    samp, _,_,_ = model.sample(1000, context=context.reshape(1,-1))
    samp = samp.squeeze()
    out_stats = out_stats+[context_state]
    samp = output_postproc(samp, out_stats)
    samp = samp.clone().detach().cpu().numpy()
    if step_ahead_max:
        return data, samp, dyna_horizon
    else:
        return data, samp, 1

def check_cuda_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return f
    

def check_dyna_fit(env, noise_params, model, replay_buffer, device, show, suffix, 
        store_dir, observe_dim, action_dim, action_dim_seq, input_preproc, output_postproc, 
        plot=True, state='random', inp_stats=None, out_stats=None, policy=[], 
        dyna_horizon=10, step_ahead_max = None):
    data, samp, dyna_horizon = sample_gt_model(env, noise_params, model, replay_buffer, device, 
        show, store_dir, observe_dim, action_dim, action_dim_seq, input_preproc, 
        output_postproc, sample=state, inp_stats=inp_stats, out_stats=out_stats, 
        policy=policy, dyna_horizon=dyna_horizon, step_ahead_max=step_ahead_max)
    scipy_kls = []
    reverse_scipy_kls = []
    for i in range(samp.shape[1]):
        #print('add evaluation metrics')
        kde_data = KernelDensity().fit(data[:,i].reshape(-1,1))
        kde_samp = KernelDensity().fit(samp[:,i].reshape(-1,1))
        points = np.linspace(data[:,i].min(), data[:,i].max(), 1000).reshape(-1,1)
        scores_data = np.exp(kde_data.score_samples(points))
        scores_samp =  np.exp(kde_samp.score_samples(points))
        scipy_kl = entropy(scores_data.reshape(-1,1), scores_samp.reshape(-1,1))[0]
        reverse_scipy_kl = entropy(scores_samp.reshape(-1,1), scores_data.reshape(-1,1))[0]
        scipy_kls.append(scipy_kl)
        reverse_scipy_kls.append(reverse_scipy_kl)
    kde=True
    if not noise_params:
        kde = False
    plot_state1(samp, data, show=show, suffix=(suffix+'_stepahead'+str(dyna_horizon))
        , store_dir=store_dir, kde=kde)
    return [scipy_kls, reverse_scipy_kls]

def check_dyna_fit_1d(env, model, replay_buffer, device, suffix, 
        store_dir, input_preproc, output_postproc, 
        plot=True, state='random', inp_stats=None, out_stats=None):
    x = np.random.choice(replay_buffer[0])
    x = x.reshape(-1,1)
    x = torch.tensor(x, dtype = torch.float32).to(device)
    x = input_preproc(x, model.stats_inputs)
    samp = model.sample(10000, context = x.to(device))
    samp = samp[0]
    samp = samp.squeeze(0)
    samp = output_postproc(samp, model.stats_outputs)
    samp = samp.detach().cpu()
    x = x.cpu()
    if env == 'bimodal':
        gt = bimodal_samp_condition(10000, x)
    else:
        gt = hetero_samp_condition(10000,x)
    gt = numpy.expand_dims(gt, axis=1)
    plot_state1(samp, gt, show=False, suffix=suffix, store_dir=store_dir, kde=True)

def sample_trajectory_replay_buffer(memory, horizon, observe_dim, action_dim):
    state, action, reward, next_state, done, noisy_actions, indeces = \
    map(np.stack, zip(*memory.buffer))
    (split_state, split_next_state, split_action,
        split_done, split_reward, split_noisy_actions, split_indices) = \
        memory.episode_split(state, action,
        reward, next_state, done, noisy_actions, indeces)
    split_state = [i for i in split_state if i.shape[0]!=0]
    split_action = [i for i in split_action if i.shape[0]!=0]
    episode_number = np.random.randint(len(split_state))
    episode_length = split_state[episode_number].shape[0]
    if episode_length > horizon:
        step_number = np.random.randint((episode_length-horizon))
    else:
        step_number = 0
    start_state = split_state[episode_number][step_number, :]
    actions = split_action[episode_number][step_number:(step_number+horizon), :]
    if actions.shape == (1,0):
        empty_actions = [[] for i in range(horizon)]
        actions = torch.tensor(empty_actions).type(torch.float32)
        actions = actions.T.unsqueeze(0)
    else:
        actions = torch.tensor(actions.reshape(1,action_dim,horizon)).type(torch.float32)
    return start_state, actions
    
def sample_trajectories(env, noise_params, dyna_model, observe_dim, action_dim, 
    input_preproc, output_postproc, particle_numb, horizon, start_state, actions, 
    inp_stats, out_stats, device):
    context_state = env.reset(state=start_state)
    samp = np.zeros([particle_numb, observe_dim, horizon])
    axis = 0
    for i in range(horizon):
        check_action = actions[i,:].numpy()
        if check_action.shape ==():
            check_action = np.expand_dims(check_action,axis=0)
        if i>0:
            check_action = np.tile(check_action, [cur_samp.shape[0], 1])
            context_state = cur_samp
            axis = 1
        context = torch.tensor(np.concatenate([context_state, check_action], axis=axis))\
                .type(torch.float32).to(device)
        context = input_preproc(context, inp_stats)
        #if i>0:
        #    not_nans = ~context.isnan().any(1)
        #    context = context[not_nans, :]
        #    samp = samp[not_nans.cpu(),:,:]
        #    not_infs = ~context.isinf().any(1)
        #    context = context[not_infs, :]
        #    samp = samp[not_infs.cpu(),:,:]
        if i == 0:
            cur_samp = dyna_model.sample(particle_numb, context=context.reshape(1,-1))
        else:
            context = torch.clamp(context, min=0, max=1)
            cur_samp = dyna_model.sample(1, context=context)
        cur_samp = cur_samp.squeeze()
        out_stats = out_stats+[context_state]
        cur_samp = output_postproc(cur_samp, out_stats)
        not_nans = ~cur_samp.isnan().any(1)
        cur_samp = cur_samp[not_nans, :]
        samp = samp[not_nans.cpu(),:,:]
        not_infs = ~cur_samp.isinf().any(1)
        cur_samp = cur_samp[not_infs, :]
        samp = samp[not_infs.cpu(),:,:]
        cur_samp = cur_samp.clone().detach().cpu().numpy()
        samp[:, :, i] = cur_samp
    actions_list = [actions[i,:].numpy() for i in range(actions.shape[0])]
    ground_truth, _, __ = sim_env(env, particle_numb, noise_params, dyna_horizon = horizon, 
                           gym_state = start_state,  actions = actions_list, 
                           sample = True, store=False, rand_init=False)
    if env.unwrapped.spec.id[-2:] =='v2':
        ground_truth = ground_truth[:,1:,:]
        samp = samp[:,1:,:]
    return ground_truth, samp

def analyze_fitted_model(env, noise_params, model, replay_buffer, device, show,
    store_dir, observe_dim, action_dim, action_dim_seq, input_preproc, output_postproc,
    plot=True, sample='random', inp_stats=None, out_stats=None, policy=[],
    dyna_horizon=10, noisy_state = False, pairs = [], rc_data=False):
    stats_list = []
    raw_nbs_list = []
    for i in range(len(pairs)):
        if rc_data:
            context = torch.tensor(np.zeros(8)).type(torch.float32).to(device)
            context = input_preproc(context, inp_stats)
            samp = model.sample(1000, context=context.reshape(1,-1))
            samp = output_postproc(samp, out_stats)
            model_pred = samp.clone().detach().cpu().numpy().squeeze()
            data = [i for i in replay_buffer.buffer if (i[0] == np.zeros(8)).all()]
            data = [i[3] for i in data]
            gt = np.stack(data)
        else:
            state = pairs[i][0]
            action = pairs[i][1] 
            gt, model_pred, _ = sample_gt_model(env, noise_params, model, replay_buffer, device, 
                show, store_dir, observe_dim, action_dim, action_dim_seq, input_preproc, 
                output_postproc, sample=sample, inp_stats=inp_stats, out_stats=out_stats, 
                policy=policy, dyna_horizon=dyna_horizon, noisy_state = noisy_state, state=state, 
                action=action)
            if env.unwrapped.spec.id[-2:] =='v2':
                gt = gt[:,1:]
                model_pred = model_pred[:,1:]
        stats, raw_nbs = get_stats(gt, model_pred)
        stats_list.append(stats)
        raw_nbs_list.append(raw_nbs)
    df = pd.DataFrame(stats_list)
    df = df.replace([np.inf, -np.inf], np.nan)
    stats = df.mean().to_dict()
    forward_kls = [i['Forward KLs']for i in raw_nbs_list]
    forward_kls = np.array(forward_kls)
    forward_kls[forward_kls==np.inf]=np.nan
    forward_kls[forward_kls==np.NINF]=np.nan
    forward_kls = np.nanmean(forward_kls, axis=0)
    return stats, raw_nbs_list, forward_kls


def get_stats(ground_truth, prediction):
    forward_kls = []
    backward_kls = []
    for i in range(ground_truth.shape[1]):
        gt_dim = ground_truth[:,i]
        pred_dim = prediction[:,i]
        forward_kl = naive_estimator(np.expand_dims(gt_dim, axis=1), 
                                        np.expand_dims(pred_dim, axis=1))
        backward_kl = naive_estimator(np.expand_dims(pred_dim, axis=1), 
                                        np.expand_dims(gt_dim, axis=1))
        forward_kls.append(forward_kl)
        backward_kls.append(backward_kl)
    forward_kl = naive_estimator(ground_truth, prediction)
    backward_kl = naive_estimator(prediction, ground_truth)
    stat_dict = {'Forward KL': forward_kl, 
            'Backward KL': backward_kl}
    raw_numbs = {'Forward KLs': forward_kls,
            'Backward KLs': backward_kls, 'Forward KL': forward_kl, 
            'Backward KL': backward_kl}
    return stat_dict, raw_numbs

def get_traj_stats(ground_truth, prediction):
    forward_kls_mean_dict = {}
    backward_kls_mean_dict = {}
    forward_kls_std_dict = {}
    backward_kls_std_dict = {}
    keys = ['nflows', 'nflows_rqs', 'MDN', 'PE']
    for key in keys:
        model_pred = prediction[key] 
        n_steps = model_pred[0].shape[2]
        forward_kls_mean_list = []
        backward_kls_mean_list = []
        forward_kls_std_list = []
        backward_kls_std_list = []
        print(key)
        for i in range(n_steps):
            pred_n = [p[:,:,i] for p in model_pred]
            gt_n = [p[:,:,i] for p in ground_truth]
            forward_kls = [naive_estimator(gt_n[s], pred_n[s]) for s in range(len(pred_n))]
            backward_kls = [naive_estimator(pred_n[s], gt_n[s]) for s in range(len(pred_n))]
            forward_kls = np.array(forward_kls)
            forward_kls[forward_kls==np.inf]=np.nan
            forward_kls[forward_kls==np.NINF]=np.nan
            forward_kls_mean = np.nanmean(forward_kls, axis=0)
            forward_kls_std = np.nanstd(forward_kls, axis=0)
            backward_kls = np.array(backward_kls)
            backward_kls[backward_kls==np.inf]=np.nan
            backward_kls[backward_kls==np.NINF]=np.nan
            backward_kls_mean = np.nanmean(backward_kls, axis=0)
            backward_kls_std = np.nanstd(backward_kls, axis=0)
            forward_kls_mean_list.append(forward_kls_mean)
            forward_kls_std_list.append(forward_kls_std)
            backward_kls_mean_list.append(backward_kls_mean)
            backward_kls_std_list.append(backward_kls_std)
        forward_kls_mean_dict[key] = forward_kls_mean_list
        forward_kls_std_dict[key] = forward_kls_std_list
        backward_kls_mean_dict[key] = backward_kls_mean_list
        backward_kls_std_dict[key] = backward_kls_std_list
    return (forward_kls_mean_dict, forward_kls_std_dict, 
        backward_kls_mean_dict, backward_kls_std_dict)


def plot_kl_trajectory(means, stds, store_dir, title):
    keys = ['nflows', 'nflows_rqs', 'MDN', 'PE'] 
    tick_size = 12
    axes_size = 16
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    plt.rc('axes', labelsize=axes_size)
    for key in keys:
        x = np.linspace(1, len(means[key]), len(means[key]))
        if key =='nflows':
            label = 'NF_cs'
        elif key =='nflows_rqs':
            label = 'NF_rqs'
        else:
            label = key
        plt.plot(x, means[key], label=label)
        plt.fill_between(x, np.array(means[key])-np.array(stds[key]), 
            np.array(means[key])+np.array(stds[key]), alpha=0.5)
    plt.legend(loc = 'upper right', fontsize='large', edgecolor='0.5')
    plt.xlabel("Horizon")
    plt.ylabel("KL Divergence")
    file_name = os.path.join(store_dir, (title+'.png'))
    plt.savefig(file_name)
    plt.close()

def plot_one_step(trajs, store_dir):
    cols = 3
    tick_size = 14
    axes_size = 18
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    plt.rc('axes', labelsize=axes_size)
    fig, axes = plt.subplots(sharex='row', ncols=5, nrows=cols, 
                                figsize=(20, 10), constrained_layout=True)
    j = 0 
    models = []
    keys = ['gt', 'nflows', 'nflows_rqs', 'MDN', 'PE']
    random_dims = np.random.choice(trajs['gt'].shape[1],size=3,replace=False)
    random_dims.sort()
    for key in keys:
        samp = trajs[key]
        nan_trajectories = np.isnan(samp).any((1,2))
        samp = samp[~nan_trajectories,:,:]
        if samp.shape[1]==0:
            print('no trajectory graphs')
            return
        dims = []
        models.append(key)
        for i in range(cols):
            dim = random_dims[i]
            sns.histplot(samp[:,dim].squeeze(), kde=True,
                 bins = 60, color = 'skyblue',
                 edgecolor = 'black', stat = 'density',
                 label = 'sampled_model', ax = axes[i,j])
        j += 1
    for ax, col in zip(axes[0], models):
        if col =='nflows':
            col = 'NF_cs'
        if col =='nflows_rqs':
            col = 'NF_rqs'
        if col =='gt':
            col = 'GT'
        ax.set_title(col, size=axes_size)
    for ax, row in zip(axes[:,0], random_dims):
        ax.set_ylabel(f'Dim {row}')
    for ax in axes[:,1]:
        ax.set_ylabel('')
    for ax in axes[:,2]:
        ax.set_ylabel('')
    for ax in axes[:,3]:
        ax.set_ylabel('')
    for ax in axes[:,4]:
        ax.set_ylabel('')
    fig.tight_layout()
    file_name = os.path.join(store_dir, ('one_step_histograms.png'))
    plt.savefig(file_name)
    plt.close()


def calc_rmse(test_data, input_preproc, output_postproc, 
        dyna_model, gp=False, ensemble=False, ensemble_size =10,
        device = 'cuda'):
    batch = [tpl for tpl in test_data.buffer]
    states, actions, reward, next_states, done, noisy_actions, index = map(np.stack, zip(*batch))
    states = torch.tensor(states, dtype = torch.float32).to(device)
    actions = torch.tensor(actions, dtype = torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype = torch.float32).to(device)
    inps = torch.hstack([states, actions])
    inps = input_preproc(inps, dyna_model.stats_inputs)
    if gp:
        samp = dyna_model.sample(1000, inps)
        samp = samp[0]
    elif ensemble:
        numb_samps = np.random.choice(ensemble_size, size=100)
        samp = []
        for i in range(ensemble_size):
            kwargs = {'rand_mask': False, 'mask_index': i}
            comp_samp = dyna_model.sample(int((numb_samps==i).sum()), context = inps, kwargs=kwargs)
            samp += [comp_samp[0].detach().cpu()]
            del comp_samp
            gc.collect()
            torch.cuda.empty_cache()
        samp = torch.hstack(samp)
        samp = samp.squeeze()
    else:
        chunk_size = 100
        chunks = int(np.ceil(1000/chunk_size))
        samp = []
        for i in range(chunks):
            samp_ch = dyna_model.sample(chunk_size, context = inps)
            samp += [samp_ch[0].detach().cpu()]
            del samp_ch
            gc.collect()
            torch.cuda.empty_cache()
        samp = torch.hstack(samp)
        samp = samp.squeeze()
    rmses = []
    for i in range(next_states.shape[0]):
        y_hat = output_postproc(samp[i,:], [i.cpu() for i in dyna_model.stats_outputs])
        y_gt = next_states[i,:].cpu()
        rmse_pt = (torch.sqrt((y_gt - y_hat)**2)).mean()
        rmses.append(rmse_pt.item())
    rmse_mean = np.mean(rmses)
    return rmse_mean

def calc_rmse_1d(test_data, input_preproc, output_postproc, 
        dyna_model, gp=False, ensemble=False, ensemble_size =10,
        device = 'cuda'):
    X = test_data[0].reshape(-1,1)
    X = torch.tensor(X, dtype = torch.float32).to(device)
    X = input_preproc(X, dyna_model.stats_inputs)
    y = test_data[1]
    if gp:
        samp = dyna_model.sample(1000, X)
        samp = samp[0]
    elif ensemble:
        numb_samps = np.random.choice(ensemble_size, size=100)
        #numb_samps = np.random.choice(ensemble_size, size=1000)
        samp = []
        torch.cuda.empty_cache()
        for i in range(ensemble_size):
            kwargs = {'rand_mask': False, 'mask_index': i}
            comp_samp = dyna_model.sample(int((numb_samps==i).sum()), context = X, kwargs=kwargs)
            samp += [comp_samp[0].detach().cpu()]
            del comp_samp
            gc.collect()
            torch.cuda.empty_cache()
        samp = torch.hstack(samp)
    else:
        chunk_size = 100
        chunks = int(np.ceil(1000/chunk_size))
        samp = []
        for i in range(chunks):
            samp_ch = dyna_model.sample(chunk_size, context = X)
            samp += [samp_ch[0].detach().cpu()]
            del samp_ch
            gc.collect()
            torch.cuda.empty_cache()
        samp = torch.hstack(samp)
    samp = samp.squeeze()
    rmses = []
    for i in range(y.shape[0]):
        y_hat = output_postproc(samp[i,:].reshape(-1,1), dyna_model.stats_outputs)
        y_gt = y[i]
        rmse_pt = (torch.sqrt((y_gt - y_hat)**2)).mean()
        rmses.append(rmse_pt.item())
    rmse_mean = np.mean(rmses)
    return rmse_mean

def calc_kl(dyna_model, context, input_preproc, output_postproc, gt_sampler):
    input_xs = input_preproc(context, dyna_model.stats_inputs)
    model_samp, _, _,_ = dyna_model.sample(1000, input_xs)
    mdl_kls = []
    for i in range(model_samp.shape[0]):
        gt_samp = gt_sampler(1000, torch.tensor(context[i,:]).cpu())
        model_out = (output_postproc(model_samp[i,:,:], dyna_model.stats_outputs)).cpu().numpy()
        kl = naive_estimator(gt_samp.reshape(1000,-1), model_out)
        mdl_kls.append(kl)
    return mdl_kls

def calc_kl_multid(dyna_model, context, input_preproc, output_postproc, env, 
        noise_params, policy, actions, states, dyna_horizon):
    input_xs = input_preproc(context, dyna_model.stats_inputs)
    model_samp, _, _,_ = dyna_model.sample(1000, input_xs)
    mdl_kls = []
    for i in range(model_samp.shape[0]):
        gt_samp,_,_ = sim_env(env, 1000, noise_params, rand_init=False, policy=policy, 
            gym_state = states[i, :], actions = actions[i, :], sample=True, 
            dyna_horizon = dyna_horizon, store=False)
        gt_samp = gt_samp[:,:,-1]
        model_out = (output_postproc(model_samp[i,:,:], dyna_model.stats_outputs)).cpu().numpy()
        kl = naive_estimator(gt_samp, model_out)
        mdl_kls.append(kl)
    return mdl_kls
