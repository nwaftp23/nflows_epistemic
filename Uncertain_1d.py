import json
import os
import argparse
import sys
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import imageio

from plot_functions import plot_input_distribution, plot_rmse_likelihood
from analyze_fit import check_dyna_fit_1d, calc_rmse_1d, calc_kl
from estimate_uncertainty import check_uncertainty_1d, find_best_points_1d
from utils import (instantiate_model, normalize, un_normalize, 
    identity, gen_folder_uncertain, seed_everything, natural_keys)
from envs_1d import (hetero_samp, hetero_samp_unif, bimodal_samp, bimodal_samp_unif,
    hetero_samp_condition, bimodal_samp_condition, bimodal_log_likelihood)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="hetero",
                        help='Environment [bimodal, hetero]')
    parser.add_argument('--seed', type=int, default=1456,
                        help='random seed (default: 123456)')
    parser.add_argument('--num_layers', default=3, help='total number of flows', type = int)
    parser.add_argument('--hids', type = int, default = 256, help='hidden units in flows')
    parser.add_argument('--lr', default=5e-4, type=float, help='flows learning rate')
    parser.add_argument('--gamma', default=0.999, type=float, help='schedule for lr step')
    parser.add_argument('--batch_size', default=256, type=int, help='size of training batch size')
    parser.add_argument('--bins', type = int, default = 10, help='number of bins for spline NSF')
    parser.add_argument('--domain', type = float, default = 4, help='domain for spline NSF')
    parser.add_argument('--show', action= 'store_true', help='show graphs')
    parser.add_argument('--process_data_output', type=str, default= 'normalize', 
            help='how to process out outputs for model')
    parser.add_argument('--process_data_input', type=str, default= 'normalize', 
            help='how to process out intputs for model')
    parser.add_argument('--epochs', default=100, type=int, 
            help='number of epochs for model')
    parser.add_argument('--model', default="nflows_ensemble", type =str,
            help='Selects the model [nflows, gp, mc_drop, nn_ensemble, nflows_ensemble])')
    parser.add_argument('--ensemble_size', default=5, type = int,
            help='number of components in uncertainty models')
    parser.add_argument('--epochs_multiplier', type=int, default=100,
                        help='number of printouts')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--data_size', type=int, default=-4,
                        help='controls size of the data (negative number use all data)')
    parser.add_argument('--action_seq', action='store_true', 
                        help='sequence of actions to predict next state')
    parser.add_argument('--noisy_state', action="store_true",
                        help='noise on state or action')
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
    parser.add_argument('--index', type=int, default=-50,
                        help='Index for hyperparam list')
    parser.add_argument('--test_acquisition', action="store_true", 
                        help='test different acquisitions')
    parser.add_argument('--acquisition_type', type=str, default='mutual_info_base',
                        help='how to acquire new points')
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)
    store_dir = './results'
    save_model_dir = './models'
    output_preproc = normalize
    output_postproc = un_normalize
    input_preproc = normalize
    input_postproc = un_normalize
    if args.model == 'gp':
        output_preproc = identity
        output_postproc = identity
        input_preproc = identity
        input_postproc = identity
    branch_folder, child_folder = gen_folder_uncertain(args)
    store_dir = os.path.join(store_dir, branch_folder)
    store_dir = os.path.join(store_dir, child_folder)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    save_model_dir  = os.path.join(save_model_dir, branch_folder)
    save_model_dir  = os.path.join(save_model_dir, child_folder)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.env == 'bimodal':
        train_data = bimodal_samp(100)  
        test_data = bimodal_samp_unif(20000)
        oracle_data = bimodal_samp(100000)
        gt_sampler = bimodal_samp_condition
    elif args.env == 'hetero':
        train_data = hetero_samp(100)  
        test_data = hetero_samp_unif(20000)
        oracle_data = hetero_samp(100000)
        gt_sampler = hetero_samp_condition
    context_dim = 1
    output_dim = 1
    args.output_dim = output_dim
    args.context_dim = context_dim
    with open(os.path.join(store_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(save_model_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    model = instantiate_model(args, output_dim, context_dim, device, input_preproc,
        output_preproc, 2)
    test_losses = []
    rmses = []
    kls = []
    numb_points_2_add = 10
    gp = False
    ensemble = False
    if args.model == 'nn_ensemble':
        numb_samps = 5000
        uncertainty_type = 'ensemble'
        ensemble = True
    elif args.model == 'mc_drop':
        numb_samps = 2500
        uncertainty_type = 'ensemble'
        ensemble = True
    elif args.model == 'nflows_ensemble':
        numb_samps = 5000
        ensemble = True
        if not args.uncertain_nflows:
            uncertainty_type = 'nflows_ensemble'
            numb_samps = 1000
        else:
            uncertainty_type = 'nflows_ensemble_out'
    elif args.model == 'nflows':
        numb_samps = 1000
        uncertainty_type = 'nflows'
    elif args.model == 'gp':
        numb_samps = 100
        uncertainty_type = 'gp'
        gp = True
    ## Test
    numb_samps = 10
    ##
    train_set_size = [len(train_data[0])]
    for i in range(args.epochs_multiplier):
        start_time = time.time()
        train_loss = model.train_1d(args.epochs, train_data, output_postproc)
        model.detach_model()
        test_loss = model.loss_1d(test_data)
        epoch_suffix = 'epoch_'+str(((i+1)))
        check_uncertainty_1d(model, imgs_dir, input_preproc,
            args.ensemble_size, device, epoch_suffix, numb_samps, 
            uncertainty=uncertainty_type, env=args.env)
        check_dyna_fit_1d(args.env, model, test_data, device,  
            epoch_suffix, imgs_dir, input_preproc, 
            output_postproc, inp_stats = model.stats_inputs,
            out_stats = model.stats_outputs)
        idxs = np.random.choice(oracle_data[0].shape[0], 1000, replace=False)
        samp_oracle = (oracle_data[0][idxs], oracle_data[1][idxs]) 
        if args.acquisition_type != 'random':
            points_2_add = find_best_points_1d(samp_oracle, uncertainty_type, int(numb_samps/5), 
                model, input_preproc, args.ensemble_size, 
                device, acquisition_criteria = args.acquisition_type)
        else:
            idxs = np.random.choice(samp_oracle[0].shape[0], 10, replace=False)
            points_2_add = (samp_oracle[0][idxs], samp_oracle[1][idxs])
            torch.cuda.empty_cache()
        train_data = (np.concatenate([train_data[0], points_2_add[0]]),
            np.concatenate([train_data[1], points_2_add[1]]))
        rmse = calc_rmse_1d(test_data, input_preproc, output_postproc, 
            model, gp = gp, ensemble = ensemble, 
            ensemble_size = args.ensemble_size, device = device)
        test_points = np.random.choice(test_data[0], size=100, replace=False)
        test_points = torch.tensor(test_points.reshape(-1,1), dtype=torch.float32).to(device)
        kl = calc_kl(model, test_points, input_preproc, output_postproc, gt_sampler)
        train_losses = train_loss
        test_losses += [test_loss]
        rmses.append(rmse)
        kls.append(np.mean(kl))
        mean_loss = torch.tensor(train_loss).mean() 
        test_likelihood = np.exp(-np.array(test_losses))
        plot_rmse_likelihood(train_losses, np.arange(len(train_losses)), 
            'train_loss', store_dir=results_dir)
        plot_rmse_likelihood(test_losses, train_set_size, 
            'test_loss', store_dir=results_dir)
        plot_rmse_likelihood(rmses, train_set_size, 'rmse', store_dir=results_dir)
        plot_rmse_likelihood(kls, train_set_size, 'kl', store_dir=results_dir)
        plot_rmse_likelihood(test_likelihood, train_set_size, 
            'likelihood', store_dir=results_dir)
        train_set_size.append(len(train_data[0]))
        end_time = time.time()
        train_time = str(timedelta(seconds=(end_time-start_time)))
        performance_string = f'Epoch: {i}, '\
                             f'Train Loss: {mean_loss:.2f}, '\
                             f'test Loss: {test_loss:.2f}, '\
                             f'Train Time: {train_time}'
        print(performance_string)
        print(f'RMSE Test: {rmse:.2f}, KL Test: {np.mean(kl):.2f},'\
            f' Train Set Size: {len(train_data[0])-numb_points_2_add}')
        np.save(os.path.join(results_dir, ('train_loss_array')), np.array(train_losses))
        np.save(os.path.join(results_dir, ('test_loss_array')), np.array(test_losses))
        np.save(os.path.join(results_dir, ('rmse_array')), np.array(rmses))
        np.save(os.path.join(results_dir, ('kls_array')), np.array(kls))
        print('Saving Model')
        model_path = os.path.join(save_model_dir,('model.pt'))
        model.save_model(model_path)
        model = instantiate_model(args, output_dim, context_dim, device, input_preproc,
            output_preproc, 2)
        print("-----------------------------------------------")
