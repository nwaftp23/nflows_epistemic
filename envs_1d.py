import torch
import numpy as np

def heteroscedastic(x):
    return 7*np.sin(x)+3*np.abs(np.cos(x/2))*np.random.normal(size=x.shape)

def hetero_samp(num_samps):
    comp = np.random.choice(3, num_samps)
    means = [-4, 0, 4]
    stds = [2/5,9/10,2/5]
    xs = np.zeros(num_samps)
    for i in range(3):
        xs[i==comp] = np.random.normal(means[i], stds[i], (i==comp).sum())
    ys = heteroscedastic(xs)
    return xs, ys

def hetero_samp_condition(num_samps, x):
    xs = x.numpy()
    xs = xs.repeat(num_samps)
    ys = heteroscedastic(xs)
    return ys

def hetero_samp_unif(num_samps):
    domain = [-4.5, 4.5]
    xs = np.random.uniform(domain[0], domain[1], num_samps)
    ys = heteroscedastic(xs)
    return xs, ys


def bimodal_1(xs):
    return 10*np.sin(xs)+np.random.normal(size=xs.shape)

def bimodal_2(xs):
    return 10*np.cos(xs)+np.random.normal(size=xs.shape)+(20-xs)
    

def bimodal_samp(num_samps):
    lamb = 2
    xs = np.random.exponential(scale = 1/lamb, size=num_samps)
    ys = np.zeros(num_samps)
    comp = np.random.choice(2, xs.shape[0])
    for i in range(2):
        if i == 0:
            ys[comp==0] = bimodal_1(xs[comp==0])
        elif i == 1:
            ys[comp==1] = bimodal_2(xs[comp==1])
    return xs, ys

def bimodal_samp_condition(num_samps, x):
    xs = x.numpy()
    xs = xs.repeat(num_samps)
    ys = np.zeros(num_samps)
    comp = np.random.choice(2, xs.shape[0])
    for i in range(2):
        if i == 0:
            ys[comp==0] = bimodal_1(xs[comp==0])
        elif i == 1:
            ys[comp==1] = bimodal_2(xs[comp==1])
    return ys

def bimodal_samp_unif(num_samps):
    domain = [0, 4]
    xs = np.random.uniform(domain[0], domain[1], num_samps)
    ys = np.zeros(num_samps)
    comp = np.random.choice(2, xs.shape[0])
    for i in range(2):
        if i == 0:
            ys[comp==0] = bimodal_1(xs[comp==0])
        elif i == 1:
            ys[comp==1] = bimodal_2(xs[comp==1])
    return xs, ys


def bimodal_log_likelihood(samp):
    xs = samp[0].reshape(-1,1)
    ys = samp[1].reshape(-1,1)
    mu1 = 10*np.cos(xs)+(20-xs)
    mu2 = 10*np.sin(xs)
    mode_1 = torch.distributions.normal.Normal(torch.tensor(mu1), torch.tensor([1.0]))
    mode_2 = torch.distributions.normal.Normal(torch.tensor(mu2), torch.tensor([1.0]))
    log_prob_1 = mode_1.log_prob(torch.tensor(ys))
    log_prob_2 = mode_2.log_prob(torch.tensor(ys))
    log_prob = torch.log(torch.exp(log_prob_1)*1/2+torch.exp(log_prob_2)*1/2)
    return log_prob

