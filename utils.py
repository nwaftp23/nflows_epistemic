import re
import random
import os
import sys
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import numpy.linalg as la
from numpy import log
import scipy
from scipy.stats import norm, rv_continuous
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree

from replay_buffer import ReplayMemory, load_mem
from policy import load_policy
from nflows_ensemble_model import nflows_ensemble_model
from nflows_model import nflows_model
from mc_drop_model import mc_drop_model
from gpytorch_model import gpytorch_model
from nn_ensemble_model import nn_ensemble_model


### CONTINUOUS ESTIMATORS


def entropy_utils(x, k=3, base=np.e):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')
###

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class LogitNormal(rv_continuous):
    def __init__(self, scale=1, loc=0):
        super().__init__(self)
        self.scale = scale
        self.loc = loc

    def _pdf(self, x):
        return norm.pdf(logit(x), loc=self.loc, scale=self.scale)/(x*(1-x))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 


def split_mean_kls(all_kls): 
    num_kls = len(all_kls[0]) 
    reformated = [] 
    means = [] 
    for i in range(num_kls): 
        reformed = [np.array(j[i]) for j in all_kls] 
        reformed = np.array(reformed) 
        reformated.append(reformed) 
        mean = reformed[np.isfinite(reformed).any(axis=1)].mean(0) 
        means.append(mean) 
    return reformated, means 

def gen_folder_uncertain(args):
    branch_folder = args.env
    sub_branch_folder = args.model
    if sub_branch_folder in ['nflows_ensemble', 'nn_ensemble']:
        if args.multihead:
            sub_branch_folder += '_multihead'
        if args.dropout_masks:
            sub_branch_folder += '_fixedmasks'
        if args.base_distro:
            sub_branch_folder += '_base'
        if args.uncertain_nflows:
            sub_branch_folder += '_flows'
    if args.test_acquisition:
        branch_folder += '_test_aquisition'
        sub_branch_folder += '_'+args.acquisition_type
    sub_branch_folder += '_seed'+str(args.seed)
    return branch_folder, sub_branch_folder

def state_diff(next_states, stats):
    print('state diff does not work when you analyze fit')
    return next_states-stats[-1]

def normalize(states, stats):
    return (states-stats[0])/(stats[1]-stats[0])

def standardize(states, stats):
    return (states-stats[2])/(stats[3])

def un_state_diff(preds, stats):
    return preds+stats[-1]

def un_normalize(preds, stats):
    return preds*(stats[1]-stats[0])+stats[0]
    
def un_standardize(preds, stats):
    return preds*(stats[3])+stats[2]

def identity(preds, stats):
    return preds

def instantiate_model(args, output_dim, context_dim, device, 
        input_preproc, output_preproc, step_ahead_max):
    if args.model == 'nflows_ensemble':
        model = nflows_ensemble_model(args.num_layers, args.hids, output_dim, context_dim, 
                args.bins, args.domain, args.lr, args.gamma, device,
                input_preproc, output_preproc, 
                conditional_step = args.conditional_step, 
                step_ahead_max = step_ahead_max, rqs = args.rqs, base = args.base_distro, 
                flows =  args.uncertain_nflows, multihead= args.multihead, 
                fixed_masks = args.dropout_masks, ensemble_size=args.ensemble_size)
    elif args.model == 'nflows':
        model = nflows_model(args.num_layers, args.hids, output_dim, context_dim, 
                args.bins, args.domain, args.lr, args.gamma, device,
                input_preproc, output_preproc, 
                conditional_step = args.conditional_step, 
                step_ahead_max = step_ahead_max, rqs = args.rqs)
    elif args.model == 'mc_drop':
        model = mc_drop_model(args.num_layers, args.hids, output_dim, context_dim,
            args.lr, args.gamma, device, input_preproc, output_preproc, 
            conditional_step=args.conditional_step, step_ahead_max=step_ahead_max)
    elif args.model == 'nn_ensemble':
        model = nn_ensemble_model(args.num_layers, args.hids, output_dim, context_dim,
            args.lr, args.gamma, device, input_preproc, output_preproc, 
            conditional_step=args.conditional_step, step_ahead_max=step_ahead_max,
            multihead=args.multihead, fixed_masks = args.dropout_masks, 
            ensemble_size = args.ensemble_size)
    elif args.model == 'gp':
        model = gpytorch_model(input_preproc, output_preproc, device=device,
            conditional_step=args.conditional_step, step_ahead_max=step_ahead_max)
    return model

