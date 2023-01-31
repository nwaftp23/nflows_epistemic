import os
import random
import math

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def plot_noise(component_weights, alphas, betas, show = False, store_dir='yo', suffix='yoyo'):
    components = np.random.choice(len(component_weights), 10000, p=component_weights)
    samples = []
    for i in range(len(component_weights)):
        samp_i = expit(np.random.normal(alphas[i], betas[i], (components==i).sum()))*2-1
        samples.append(samp_i)
    samples = np.concatenate(samples)
    sns.displot(samples, kde=True, color = 'skyblue',
            edgecolor='black',stat = 'density', label = 'noise_distro')
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('noise_distribution_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_valley(component_weights, alphas, betas, show = False, store_dir='yo', suffix='yoyo'):
    components = np.random.choice(len(component_weights), 10000, p=component_weights)
    samples = []
    for i in range(len(component_weights)):
        samp_i = np.random.beta(alphas[i], betas[i], (components==i).sum())*2-1
        samples.append(samp_i)
    samples = np.concatenate(samples)
    sns.displot(samples, kde=True, color = 'skyblue',
            edgecolor='black',stat = 'density', label = 'noise_distro')
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('noise_distribution_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

