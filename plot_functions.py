import re
from copy import copy
import random
import os
import sys

#sys.stderr = object
#sys.tracebacklimit=0
import torch
import torchvision
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.special import logit, expit
from scipy.stats import norm, rv_continuous, entropy, wasserstein_distance
from sklearn.neighbors import KernelDensity
from scipy.stats.kde import gaussian_kde

from replay_buffer import *

def get_thetas(states):
   thetas = np.arctan2(states[:,1], states[:,0])
   return thetas

def plot_noise(component_weights, alphas, betas, show = False, 
        store_dir='yo', suffix='yoyo', noise_weight = 1):
    components = np.random.choice(len(component_weights), 10000, p=component_weights)
    samples = []
    for i in range(len(component_weights)):
        samp_i = (expit(np.random.normal(alphas[i], betas[i], (components==i).sum()))*2
                -1)*noise_weight
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

def plot_valley(component_weights, alphas, betas, show = False, 
        store_dir='yo', suffix='yoyo', noise_weight = 1):
    components = np.random.choice(len(component_weights), 10000, p=component_weights)
    samples = []
    for i in range(len(component_weights)):
        samp_i = (np.random.beta(alphas[i], betas[i], (components==i).sum())*2-1)*noise_weight
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


def plot_states(thetas_train, actions_train, thetas_test, actions_test, store_dir, suffix):
    plt.scatter(thetas_train, actions_train, marker='x', label = 'train', alpha = 0.05, s=10)
    plt.scatter(thetas_test, actions_test, marker='o', label='test', alpha = 0.05, s=10)
    plt.legend(loc='upper left')
    file_name = os.path.join(store_dir, (f'thetavthetadot_'+suffix+'.png'))
    plt.savefig(file_name)
    plt.close()

def plot_likelihood_histograms(likelihoods, suffix, train_dataset, 
    store_dir = '', show = False):
    #width = len(likelihoods)*12
    #f, axes = plt.subplots(1, len(likelihoods), figsize=(width,12))
    #f, axes = plt.subplots(1, 1)
    colors = ['skyblue', 'lightcoral', 'limegreen', 'mediumpurple', 'bisque', 'sandybrown']
    index = 1
    for key, likelihood in likelihoods.items():
        likelihoods[key] = likelihoods[key].numpy()
    sns.histplot(likelihoods, kde = True, stat = 'density', log_scale=(False, False))
    #for key, likelihood in likelihoods.items():
    #    import pdb; pdb.set_trace()
    #    sns.histplot(likelihood.numpy(), kde = False, stat = 'density',
    #        label = key) #color=colors[index], edgecolor = 'black')
#            ax = axes[index])
    #    index += 1
    plt.legend()
    # title
    plt.title(f'Trained on {train_dataset}')
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('loglikeli_hist_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_state1(samp, data, show = False, suffix='jackie', store_dir='robinson', kde=True):
    width = samp.shape[1]*12
    f, axes = plt.subplots(1, samp.shape[1], sharey=False, figsize=(width,12))
    f.suptitle("Histogram of Data", fontsize=40)
    for i in range(data.shape[1]):
        # bins = 60
        if samp.shape[1] == 1:
            plot_ax = axes
        else:
            plot_ax = axes[i]
        sns.histplot(data[:,i], kde=kde, color = 'skyblue',
             edgecolor = 'black', stat = 'count',
             label = 'ground_truth', ax = plot_ax)
        sns.histplot(samp[:,i], kde=True, color = 'red',
             edgecolor = 'black', stat = 'count', 
             label = 'sampled_model', ax = plot_ax)
        plot_ax.legend(loc = 'upper right')
        plot_ax.set_title(f'Dimension {i}', fontsize=20)
    f.text(0.5, 0.04, 'Position', ha='center', fontsize=30)
    f.text(0.04, 0.5, 'Normalized Frequency', va='center', rotation='vertical', fontsize=30)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('state_distribution_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_rmse_likelihood(rmse_likelihood, x, label, show=False, store_dir='robinson'):
    plt.plot(x, rmse_likelihood)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, (f'{label}.png'))
        plt.savefig(file_name)
        plt.close()

def plot_total_ent(ent, x, store_dir, suffix, show=False):
    plt.plot(x, ent)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, (suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_state_hm(mem, test_mem, store_dir, env='Pendulum-v0', show=False):
    (train_state, train_action, reward, 
        next_state, done, noisy_action, index) = map(np.stack, zip(*mem.buffer))
    (test_state, test_action, reward, 
        next_state, done, noisy_action, index) = map(np.stack, zip(*test_mem.buffer))
    #### subset for testing
    train_state = train_state[:10000,:]
    test_state = test_state[:10000,:]
    ####
    if env == 'Pendulum-v0':
        train_thetas = get_thetas(train_state)
        test_thetas = get_thetas(test_state)
        train_state[:,0] = train_thetas
        test_state[:,0] = test_thetas
        train_state = np.delete(train_state, 1, 1)
        test_state = np.delete(test_state, 1, 1)
    f, axes = plt.subplots(2, 2, figsize=(30,30))
    axes[0,0].scatter(train_state[:,0], train_state[:,1])
    axes[0,0].set_title('Train Set')
    k = gaussian_kde(train_state.T)
    xi, yi = np.mgrid[train_state[:,0].min():train_state[:,0].max():train_state[:,0].size**0.5*1j,
        train_state[:,1].min():train_state[:,1].max():train_state[:,1].size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    pm = axes[0,1].pcolormesh(xi, yi, zi.reshape(xi.shape))
    f.colorbar(pm, ax=axes[0,1])
    axes[0,1].set_xlim(train_state[:,0].min(), train_state[:,0].max())
    axes[0,1].set_ylim(train_state[:,1].min(), train_state[:,1].max())
    axes[1,0].scatter(test_state[:,0], test_state[:,1])
    axes[1,0].set_title('Test Set')
    k = gaussian_kde(test_state.T)
    xi, yi = np.mgrid[test_state[:,0].min():test_state[:,0].max():test_state[:,0].size**0.5*1j,
        test_state[:,1].min():test_state[:,1].max():test_state[:,1].size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    pm = axes[1,1].pcolormesh(xi, yi, zi.reshape(xi.shape))
    f.colorbar(pm, ax=axes[1,1])
    axes[1,1].set_xlim(test_state[:,0].min(), test_state[:,0].max())
    axes[1,1].set_ylim(test_state[:,1].min(), test_state[:,1].max())
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('replay_buffers.png'))
        plt.savefig(file_name)
        plt.close()

def plot_hm_dep_uncertainty(total_ent, alea_ent, x_cord, y_cord, store_dir, suffix, show=False):
    epi_ent = total_ent-alea_ent
    f, axes = plt.subplots(1, 3, figsize=(45,15))
    pm = axes[0].pcolormesh(x_cord, y_cord, total_ent.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[0])
    axes[0].set_title('Total Uncertainty')
    pm = axes[1].pcolormesh(x_cord, y_cord, alea_ent.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[1])
    axes[1].set_title('Aleatoric Uncertainty')
    pm = axes[2].pcolormesh(x_cord, y_cord, epi_ent.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[2])
    axes[2].set_title('Epistemic Uncertainty')
    epoch = re.search(r'epoch_[0-9]+', suffix).group()
    epoch = epoch.split('_')[1]
    f.suptitle(f'Epoch {epoch}', fontsize=26)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('dep_uncertainty_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_total_ent_hm(total_ent, x_cord, y_cord, store_dir, suffix, show=False):
    f, axes = plt.subplots(1, 1, figsize=(15,15))
    pm = axes.pcolormesh(x_cord, y_cord, total_ent.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes)
    axes.set_title('Total Uncertainty')
    f.suptitle(suffix, fontsize=26)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('total_ent_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_dep_uncertainty_1d(total_ent, alea_ent, x_cord, store_dir, suffix, show=False):
    epi_ent = total_ent-alea_ent
    f, axes = plt.subplots(1, 3, figsize=(45,15))
    axes[0].plot(x_cord, total_ent.reshape(x_cord.shape))
    axes[0].set_title('Total Uncertainty')
    axes[1].plot(x_cord, alea_ent.reshape(x_cord.shape))
    axes[1].set_title('Aleatoric Uncertainty')
    axes[2].plot(x_cord, epi_ent.reshape(x_cord.shape))
    axes[2].set_title('Epistemic Uncertainty')
    epoch = re.search(r'epoch_[0-9]+', suffix).group()
    epoch = epoch.split('_')[1]
    f.suptitle(f'Epoch {epoch}', fontsize=26)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('dep_uncertainty_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_hm_all_uncertainty(kl_mean, kl_max, wd_mean, wd_max, dep_out, dep_base, 
    x_cord, y_cord, store_dir, suffix, show=False):
    f, axes = plt.subplots(1, 6, figsize=(90,15))
    pm = axes[0].pcolormesh(x_cord, y_cord, dep_out.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[0])
    axes[0].set_title('Dep Out')
    pm = axes[1].pcolormesh(x_cord, y_cord, dep_base.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[1])
    axes[1].set_title('Dep Base')
    pm = axes[2].pcolormesh(x_cord, y_cord, kl_mean.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[2])
    axes[2].set_title('KL mean')
    pm = axes[3].pcolormesh(x_cord, y_cord, kl_max.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[3])
    axes[3].set_title('KL max')
    pm = axes[4].pcolormesh(x_cord, y_cord, wd_mean.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[4])
    axes[4].set_title('WD mean')
    pm = axes[5].pcolormesh(x_cord, y_cord, wd_max.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[5])
    axes[5].set_title('WD max')
    epoch = re.search(r'epoch_[0-9]+', suffix).group()
    epoch = epoch.split('_')[1]
    f.suptitle(f'Epoch {epoch}', fontsize=26)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('all_uncertainty_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_hm_luc_uncertainty(kl_mean, kl_max, wd_mean, wd_max, x_cord, y_cord, 
        store_dir, suffix, show=False):
    f, axes = plt.subplots(1, 4, figsize=(60,15))
    pm = axes[0].pcolormesh(x_cord, y_cord, kl_mean.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[0])
    axes[0].set_title('KL mean')
    pm = axes[1].pcolormesh(x_cord, y_cord, kl_max.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[1])
    axes[1].set_title('KL max')
    pm = axes[2].pcolormesh(x_cord, y_cord, wd_mean.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[2])
    axes[2].set_title('WD mean')
    pm = axes[3].pcolormesh(x_cord, y_cord, wd_max.reshape(x_cord.shape))
    f.colorbar(pm, ax=axes[3])
    axes[3].set_title('WD max')
    epoch = re.search(r'epoch_[0-9]+', suffix).group()
    epoch = epoch.split('_')[1]
    f.suptitle(f'Epoch {epoch}', fontsize=26)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('luc_uncertainty_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_luc_uncertainty_1d(kl_mean, kl_max, wd_mean, wd_max, x_cord,
        store_dir, suffix, show=False):
    f, axes = plt.subplots(1, 4, figsize=(60,15))
    axes[0].plot(x_cord, kl_mean.reshape(x_cord.shape))
    axes[0].set_title('KL mean')
    axes[1].plot(x_cord, kl_max.reshape(x_cord.shape))
    axes[1].set_title('KL max')
    axes[2].plot(x_cord, wd_mean.reshape(x_cord.shape))
    axes[2].set_title('WD mean')
    axes[3].plot(x_cord, wd_max.reshape(x_cord.shape))
    axes[3].set_title('WD max')
    epoch = re.search(r'epoch_[0-9]+', suffix).group()
    epoch = epoch.split('_')[1]
    f.suptitle(f'Epoch {epoch}', fontsize=26)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('luc_uncertainty_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()
    

def plot_input_distribution(train_x, test_x, epoch, store_dir, suffix):
    sns.histplot(data = train_x, kde=True, 
             bins=60, color = 'red', 
             stat = 'probability',
             label = 'train_set')
    sns.histplot(data = test_x, kde=True,
             bins=60, color = 'skyblue',
             stat='probability',
             label = 'test_set')
    plt.legend(loc = 'upper right')
    plt.title(f'Distribution of Input Epoch {epoch}')
    file_name = os.path.join(store_dir, ('input_distribution_'+suffix+'.png'))
    plt.savefig(file_name)
    plt.close()

