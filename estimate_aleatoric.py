import os
import argparse
import json

import numpy as np
import torch

from utils import normalize, un_normalize, identity, instantiate_model
from analyze_fit import calc_kl
from envs_1d import (bimodal_samp_unif, hetero_samp_unif, 
        bimodal_samp_condition, hetero_samp_condition)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="hetero",
                        help='Environment [bimodal, hetero]')
    args = parser.parse_args()
    base_dir = '/home/nwaftp23/scratch/uncertain_nf/mujoco'
    env_dir = os.path.join(base_dir, args.env)
    run_dirs = os.listdir(env_dir)
    run_dirs = [os.path.join(env_dir, d) for d in run_dirs
        if os.path.isdir(os.path.join(env_dir, d))]
    models_types = ['gp', 'mc_drop', 'nn_ensemble', 'nflows_ensemble', 'nflows_seed']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_ahead_max = 30
    if args.env == 'bimodal':
        test_xs, _ = bimodal_samp_unif(100)
        gt_sampler = bimodal_samp_condition
    elif args.env == 'hetero':
        test_xs, _ = hetero_samp_unif(100)
        gt_sampler = hetero_samp_condition
    test_xs = test_xs.reshape(-1,1)
    input_xs = torch.tensor(test_xs, dtype = torch.float32).to(device)
    kls = {}
    for mt in models_types:
        model_dirs = [d for d in run_dirs 
            if os.path.basename(os.path.normpath(d)).startswith(mt)]
        models = []
        input_preproc = normalize
        output_postproc = un_normalize
        output_preproc = normalize
        if mt == 'gp':
            input_preproc = identity
            output_postproc = identity
            output_preproc = identity 
        mt_kls = []
        for d in model_dirs:
            model_path = os.path.join(d, 'model.pt')
            if os.path.exists(os.path.join(d, 'model_masks.pt')):
                masks_path = os.path.join(d, 'model_masks.pt')
            saved_parser = argparse.ArgumentParser()
            saved_args = saved_parser.parse_args()
            with open(os.path.join(d, 'commandline_args.txt'), 'r') as f:
                saved_args.__dict__ = json.load(f)
            dyna_model = instantiate_model(saved_args, saved_args.output_dim, 
                saved_args.context_dim, device, input_preproc, output_preproc, step_ahead_max)
            dyna_model.load_model(model_path) 
            import pdb; pdb.set_trace()
            mdl_kls = calc_kl(dyna_model, input_xs, input_preproc, output_postproc, gt_sampler)
            mt_kls.append(mdl_kls)
            import pdb; pdb.set_trace()
