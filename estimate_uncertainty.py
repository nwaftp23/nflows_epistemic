import os
import itertools
from time import time
from datetime import timedelta

import torch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

from plot_functions import (plot_hm_dep_uncertainty, 
    plot_hm_all_uncertainty, plot_dep_uncertainty_1d, 
    plot_total_ent, plot_total_ent_hm)
from kl_estimator import naive_estimator 

def find_best_points(samp, uncertainty_type, numb_points, dyna_model, 
        input_preproc, ensemble_size, device, acquisition_criteria = 'mutual_info_base'):
    state = samp[0]
    action = samp[1]
    model_inp = np.hstack([state, action])
    model_inp = torch.tensor(model_inp).type(torch.float32).to(device)
    model_inp = input_preproc(model_inp, dyna_model.stats_inputs)
    if uncertainty_type == 'gp':
        #X = model_inp.cpu().numpy()
        #mean_hat, std_hat = dyna_model.model.predict(X, return_std=True)
        #uncertainty = std_hat
        preds = dyna_model.model(model_inp)
        uncertainty = preds.variance.sqrt().detach().cpu().mean(1).numpy()
    elif uncertainty_type == 'nflows_ensemble':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 
            'mutual_info', dyna_model, ensemble_size,  
            numb_samps = numb_points, acquisition=True)
        uncertainty = uncertainty[acquisition_criteria]
    elif uncertainty_type == 'nflows_ensemble_out':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 
            'mutual_info', dyna_model, ensemble_size,  
            numb_samps = numb_points, acquisition=True)
        if acquisition_criteria == 'mutual_info_base':
            acquisition_criteria = 'mutual_info_out'
        uncertainty = uncertainty[acquisition_criteria]
    elif uncertainty_type == 'nflows':
        uncertainty = estimate_uncertainty_nflows(model_inp, dyna_model, numb_samps = numb_points)
        uncertainty = uncertainty['total_ent_base']
    elif uncertainty_type == 'ensemble':
        uncertainty = estimate_uncertainty_ensembles(model_inp, 'mutual_info', dyna_model,
            ensemble_size, numb_samps = numb_points)
        uncertainty = uncertainty['mutual_info']
    ind = np.argpartition(uncertainty, -10)[-10:]
    points_2_add = []
    for i in ind:
        point = tuple((s[i] for s in samp))
        points_2_add.append(point)
    return points_2_add

def find_best_points_1d(samp, uncertainty_type, numb_points, dyna_model, 
        input_preproc, ensemble_size, device, acquisition_criteria = 'mutual_info_base'):
    X = samp[0].reshape(-1,1)
    X = torch.tensor(X, dtype = torch.float32).to(device)
    X = input_preproc(X, dyna_model.stats_inputs)
    if uncertainty_type == 'gp':
        preds = dyna_model.model(X)
        uncertainty = preds.variance.sqrt().detach().cpu().numpy().reshape(-1)
    elif uncertainty_type == 'nflows_ensemble':
        uncertainty = estimate_uncertainty_ensemble_nflows(X, 
            'mutual_info', dyna_model, ensemble_size,  
            numb_samps = numb_points)
        uncertainty = uncertainty[acquisition_criteria]
    elif uncertainty_type == 'nflows_ensemble_out':
        uncertainty = estimate_uncertainty_ensemble_nflows(X, 
            'mutual_info', dyna_model, ensemble_size,  
            numb_samps = numb_points)
        if acquisition_criteria == 'mutual_info_base':
            acquisition_criteria = 'mutual_info_out'
        uncertainty = uncertainty[acquisition_criteria]
    elif uncertainty_type == 'nflows':
        uncertainty = estimate_uncertainty_nflows(X, dyna_model, numb_samps = numb_points)
        uncertainty = uncertainty['total_ent_base']
    elif uncertainty_type == 'ensemble':
        uncertainty = estimate_uncertainty_ensembles(X, 'mutual_info', dyna_model,
            ensemble_size, numb_samps = numb_points)
        uncertainty = uncertainty['mutual_info']
    ind = np.argpartition(uncertainty, -10)[-10:]
    points_2_add = (samp[0][ind], samp[1][ind])
    return points_2_add

def estimate_uncertainty_ensemble_nflows(model_inp, uncertainty_type, dyna_model,
        ensemble_size, numb_samps = 10000, acquisition=False):
    uncertainty = {}
    means_base = []
    stds_base = []
    samples_base = []
    samples_out = []
    component_ent_base = []
    component_ent_out = []
    output_extra_time = []
    dep_t0 = time()
    mut_info_out = []
    mut_info_base = []
    out_log_probs = []
    base_log_probs = []
    if not acquisition:
        chunk_size = 1000
    else:
        chunk_size = 1000
    for i in range(ensemble_size):
        base_distro_samps = 1
        kwargs = {'rand_mask': False, 'mask_index': i}
        if numb_samps > chunk_size:
            chunks = int(np.ceil(numb_samps/chunk_size))
            output_hat = []
            base_hat = []
            base_log_prob = []
            nflows_log_prob = []
            component_log_prob = []
            for j in range(chunks):
                (output_hat_ch, nflows_log_prob_ch, component_log_prob_ch, 
                    base_log_prob_ch, base_hat_ch, base_mean, base_std) = (
                    dyna_model.model.sample_and_log_prob(chunk_size, context = model_inp, 
                    kwargs=kwargs, ensemble = True, ensemble_size = ensemble_size))
                output_hat.append(output_hat_ch.detach().cpu().numpy())
                base_hat.append(base_hat_ch.detach().cpu().numpy())
                base_log_prob.append(base_log_prob_ch.detach().cpu().numpy())
                nflows_log_prob.append(nflows_log_prob_ch.detach().cpu().numpy())
                component_log_prob.append(component_log_prob_ch.detach().cpu().numpy())
            output_hat = np.hstack(output_hat)
            base_hat = np.hstack(base_hat)
            base_log_prob = np.hstack(base_log_prob)
            nflows_log_prob = np.hstack(nflows_log_prob)
            component_log_prob = np.hstack(component_log_prob)
            base_mean = base_mean[::chunk_size]
            base_std = base_std[::chunk_size]
        else:
            (output_hat, nflows_log_prob, component_log_prob, base_log_prob, 
                    base_hat, base_mean, base_std) = (
                    dyna_model.model.sample_and_log_prob(numb_samps, context = model_inp, 
                    kwargs=kwargs, ensemble = True, ensemble_size = ensemble_size))
            output_hat = output_hat.detach().cpu().numpy()
            base_hat = base_hat.detach().cpu().numpy()
            nflows_log_prob = nflows_log_prob.detach().cpu().numpy()
            base_log_prob = base_log_prob.detach().cpu().numpy()
            component_log_prob = component_log_prob.detach().cpu().numpy()
            base_mean = base_mean[::numb_samps]
            base_std = base_std[::numb_samps]
        base_mean = base_mean.detach().cpu().numpy()
        base_std = base_std.detach().cpu().numpy()
        means_base.append(base_mean)
        stds_base.append(base_std)
        samples_out.append(output_hat)
        samples_base.append(base_hat)
        ent = base_mean.shape[1]/2*np.log(2*np.pi*np.e)+1/2*np.log((base_std**2).prod(1))
        component_ent_base.append(ent)
        out_log_probs.append(nflows_log_prob)
        base_log_probs.append(base_log_prob)
        if uncertainty_type in ['all', 'mutual_info', 'mean_comp_ent']:
            component_ent_out.append(np.nanmean(-component_log_prob, 1))
    y_out = np.hstack(samples_out)
    y_base = np.hstack(samples_base)
    log_probs_out = np.hstack(out_log_probs) 
    log_probs_base = np.hstack(base_log_probs) 
    ent_out = []
    ent_base = []
    weights = [1/ensemble_size]*ensemble_size
    ch_base = []
    ch_out = []
    for i in range(y_out.shape[0]):
        ent_base.append(np.mean(-log_probs_base[i,np.isfinite(log_probs_base[i,:])]))
        ent_out.append(np.mean(-log_probs_out[i,np.isfinite(log_probs_out[i,:])]))
    total_ent_base = np.array(ent_base)
    total_ent_out = np.array(ent_out)
    uncertainty['total_ent_base'] = total_ent_base
    uncertainty['total_ent_out'] = total_ent_out
    if uncertainty_type != 'tot_ent':
        alea_ent_base = np.stack(component_ent_base).mean(0)
        alea_ent_out = np.stack(component_ent_out).mean(0)
        epi_ent_base = total_ent_base-alea_ent_base
        epi_ent_out = total_ent_out-alea_ent_out
        uncertainty['mutual_info_out'] = epi_ent_out
        uncertainty['mutual_info_base'] = epi_ent_base
        uncertainty['mean_comp_ent_out'] = alea_ent_out
        uncertainty['mean_comp_ent_base'] = alea_ent_base
    return uncertainty

def estimate_uncertainty_nflows(model_inp, dyna_model, numb_samps = 10000):
    uncertainty = {}
    means_base = []
    stds_base = []
    samples_base = []
    samples_out = []
    output_extra_time = []
    out_log_probs = []
    base_log_probs = []
    chunk_size = 500
    if numb_samps > chunk_size:
        chunks = int(np.ceil(numb_samps/chunk_size))
        output_hat = []
        base_hat = []
        base_log_prob = []
        nflows_log_prob = []
        for j in range(chunks):
            (output_hat_ch, nflows_log_prob_ch, component_log_prob_ch, 
                base_log_prob_ch, base_hat_ch, base_mean, base_std) = (
                dyna_model.sample_and_log_prob(chunk_size, context = model_inp))
            output_hat.append(output_hat_ch.detach().cpu().numpy())
            base_hat.append(base_hat_ch.detach().cpu().numpy())
            base_log_prob.append(base_log_prob_ch.detach().cpu().numpy())
            nflows_log_prob.append(nflows_log_prob_ch.detach().cpu().numpy())
        output_hat = np.hstack(output_hat)
        base_hat = np.hstack(base_hat)
        base_log_prob = np.hstack(base_log_prob)
        nflows_log_prob = np.hstack(nflows_log_prob)
        base_mean = base_mean[::chunk_size]
        base_std = base_std[::chunk_size]
    else:
        (output_hat, nflows_log_prob, component_log_prob, base_log_prob, 
                base_hat, base_mean, base_std) = (
                dyna_model.sample_and_log_prob(numb_samps, context = model_inp))
        output_hat = output_hat.detach().cpu().numpy()
        base_hat = base_hat.detach().cpu().numpy()
        nflows_log_prob = nflows_log_prob.detach().cpu().numpy()
        base_log_prob = base_log_prob.detach().cpu().numpy()
        base_mean = base_mean[::numb_samps]
        base_std = base_std[::numb_samps]
    base_mean = base_mean.detach().cpu().numpy()
    base_std = base_std.detach().cpu().numpy()
    means_base.append(base_mean)
    stds_base.append(base_std)
    samples_out.append(output_hat)
    samples_base.append(base_hat)
    ent = base_mean.shape[1]/2*np.log(2*np.pi*np.e)+1/2*np.log((base_std**2).prod(1))
    out_log_probs.append(nflows_log_prob)
    base_log_probs.append(base_log_prob)
    y_out = np.hstack(samples_out)
    y_base = np.hstack(samples_base)
    log_probs_out = np.hstack(out_log_probs) 
    log_probs_base = np.hstack(base_log_probs) 
    ent_out = []
    ent_base = []
    for i in range(y_out.shape[0]):
        ent_base.append(np.mean(-log_probs_base[i,np.isfinite(log_probs_base[i,:])]))
        ent_out.append(np.mean(-log_probs_out[i,np.isfinite(log_probs_out[i,:])]))
    total_ent_base = np.array(ent_base)
    total_ent_out = np.array(ent_out)
    uncertainty['total_ent_base'] = total_ent_base
    uncertainty['total_ent_out'] = total_ent_out
    return uncertainty

def estimate_uncertainty_ensembles(model_inp, uncertainty_type, dyna_model,
        mask_samps, numb_samps = 10000):
    uncertainty = {}
    means = []
    stds = []
    output = []
    component_ent = []
    output_extra_time = []
    mut_info = []
    log_probs = []
    for i in range(mask_samps):
        kwargs = {'rand_mask': False, 'mask_index': i}
        output_hat, log_prob, mask_prob, mu, sig = (
            dyna_model.sample_and_log_prob(numb_samps, context = model_inp, 
                kwargs=kwargs))
        output_hat = output_hat.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()
        mu = mu.detach().cpu().numpy()
        sig = sig.detach().cpu().numpy()
        output.append(output_hat)
        log_probs.append(log_prob)
        means.append(mu)
        stds.append(sig)
        ent = mu.shape[1]/2*np.log(2*np.pi*np.e)+1/2*np.log((sig**2).prod(1))
        component_ent.append(ent)
    output = np.vstack(output)
    log_probs_out = []
    for i in range(mask_samps):
        norm_rv = torch.distributions.normal.Normal(torch.tensor(means[i]), 
            torch.tensor(stds[i]))
        log_prob_comp = 1/mask_samps*torch.exp(norm_rv.log_prob(torch.tensor(output)).sum(2))
        log_probs_out.append(log_prob_comp)
    log_probs_out = torch.stack(log_probs_out)
    log_probs_out = log_probs_out.numpy()
    log_probs_out = np.log(log_probs_out.sum(0))
    total_ent = -log_probs_out.mean(0)
    uncertainty['total_ent'] = total_ent
    if uncertainty_type != 'tot_ent':
        alea_ent = np.stack(component_ent).mean(0)
        epi_ent = total_ent-alea_ent
        uncertainty['mutual_info'] = epi_ent
        uncertainty['mean_comp_ent'] = alea_ent
    return uncertainty

def check_uncertainty_1d(dyna_model, store_dir, input_preproc, 
        ensemble_size, device, suffix, numb_samples, uncertainty='ensemble', env='bimodal'):
    if env == 'bimodal':
        x_cord = np.linspace(0, 4, 300)
    elif env == 'hetero':
        x_cord = np.linspace(-4.5, 4.5, 300)
    model_inp = torch.tensor(x_cord).type(torch.float32).to(device)
    model_inp = model_inp.reshape(-1,1)
    model_inp = input_preproc(model_inp, dyna_model.stats_inputs)
    if uncertainty == 'nflows_ensemble':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 'all', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_dep_uncertainty_1d(uncertainty['total_ent_base'], uncertainty['mean_comp_ent_base'],
            x_cord, store_dir, 'base_'+suffix)
        plot_dep_uncertainty_1d(uncertainty['total_ent_out'], uncertainty['mean_comp_ent_out'],
            x_cord, store_dir, 'out_'+suffix)
    if uncertainty == 'nflows_ensemble_out':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 'mutual_info', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_dep_uncertainty_1d(uncertainty['total_ent_out'], uncertainty['mean_comp_ent_out'],
            x_cord, store_dir, 'out_'+suffix)
    elif uncertainty == 'ensemble':
        uncertainty = estimate_uncertainty_ensembles(model_inp, 'all', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_dep_uncertainty_1d(uncertainty['total_ent'], uncertainty['mean_comp_ent'],
            x_cord, store_dir, suffix)
    elif uncertainty == 'nflows':
        uncertainty = estimate_uncertainty_nflows(model_inp, dyna_model,
            numb_samps = numb_samples)
        plot_total_ent(uncertainty['total_ent_base'], x_cord, store_dir, 'total_ent_base_' + suffix)
        plot_total_ent(uncertainty['total_ent_out'], x_cord, store_dir, 'total_ent_out_'+suffix)
    elif uncertainty == 'gp':
        #model_inp = model_inp.cpu().numpy()
        #mean_hat, std_hat = dyna_model.model.predict(model_inp, return_std=True)
        preds = dyna_model.model(model_inp)
        std_hat = preds.variance.sqrt().detach().cpu()
        plot_total_ent(std_hat, x_cord, store_dir, 'total_ent_'+suffix)

def check_uncertainty_wet_chicken(dyna_model, store_dir, input_preproc, 
        ensemble_size, device, suffix, numb_samples, uncertainty):
    x_cord = np.linspace(0, 5, 25)
    y_cord = np.linspace(0, 5, 25)
    xx, yy = np.meshgrid(x_cord, y_cord)
    x_cord = xx.reshape(-1,1)
    y_cord = yy.reshape(-1,1)
    model_inp = np.hstack([x_cord, y_cord])
    action = np.array([[0,0]]).repeat(model_inp.shape[0], axis=0)
    model_inp = np.hstack([model_inp, action])
    model_inp = torch.tensor(model_inp).type(torch.float32).to(device)
    model_inp = input_preproc(model_inp, dyna_model.stats_inputs)
    if uncertainty == 'nflows_ensemble':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 'all', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_hm_dep_uncertainty(uncertainty['total_ent_base'], uncertainty['mean_comp_ent_base'],
            xx, yy, store_dir, 'base_'+suffix)
        plot_hm_dep_uncertainty(uncertainty['total_ent_out'], uncertainty['mean_comp_ent_out'],
            xx, yy, store_dir, 'out_'+suffix)
    elif uncertainty == 'nflows_ensemble_out':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 'mutual_info', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_hm_dep_uncertainty(uncertainty['total_ent_out'], uncertainty['mean_comp_ent_out'],
            xx, yy, store_dir, 'out_'+suffix)
    elif uncertainty == 'ensemble':
        uncertainty = estimate_uncertainty_ensembles(model_inp, 'all', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_hm_dep_uncertainty(uncertainty['total_ent'], uncertainty['mean_comp_ent'], 
            xx, yy, store_dir, suffix)
    elif uncertainty == 'nflows':
        uncertainty = estimate_uncertainty_nflows(model_inp, dyna_model,
            numb_samps = numb_samples)
        plot_total_ent_hm(uncertainty['total_ent_base'], 
            xx, yy, store_dir,'base_'+suffix)
        plot_total_ent_hm(uncertainty['total_ent_out'], 
            xx, yy, store_dir,'out_'+suffix)
    elif uncertainty == 'gp':
        #model_inp = model_inp.cpu().numpy()
        #mean_hat, std_hat = dyna_model.model.predict(model_inp, return_std=True)
        preds = dyna_model.model(model_inp)
        std_hat = preds.variance.sqrt().detach().cpu().mean(1)
        plot_total_ent_hm(std_hat, 
            xx, yy, store_dir, suffix)

    
def check_uncertainty_pendulum(dyna_model, store_dir, input_preproc, 
        ensemble_size, device, suffix, numb_samples, uncertainty):
    x_cord = np.linspace(-np.pi, np.pi, 25)
    y_cord = np.linspace(-8, 8, 25)
    xx, yy = np.meshgrid(x_cord, y_cord)
    x_cord = xx.reshape(-1,1)
    y_cord = yy.reshape(-1,1)
    model_inp = np.hstack([x_cord, y_cord])
    action = np.array([[0]]).repeat(model_inp.shape[0], axis=0)
    sin_theta = np.sin(x_cord)
    cos_theta = np.cos(x_cord)
    new_model_inp = np.zeros([model_inp.shape[0],3])
    new_model_inp[:,0]=cos_theta.squeeze()
    new_model_inp[:,1]=sin_theta.squeeze()
    new_model_inp[:,2]=model_inp[:,1]
    model_inp = np.hstack([new_model_inp, action])
    model_inp = torch.tensor(model_inp).type(torch.float32).to(device)
    model_inp = input_preproc(model_inp, dyna_model.stats_inputs)
    if uncertainty == 'nflows_ensemble':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 'all', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_hm_dep_uncertainty(uncertainty['total_ent_base'], uncertainty['mean_comp_ent_base'],
            xx, yy, store_dir, 'base_'+suffix)
        plot_hm_dep_uncertainty(uncertainty['total_ent_out'], uncertainty['mean_comp_ent_out'],
            xx, yy, store_dir, 'out_'+suffix)
    elif uncertainty == 'nflows_ensemble_out':
        uncertainty = estimate_uncertainty_ensemble_nflows(model_inp, 'mutual_info', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_hm_dep_uncertainty(uncertainty['total_ent_out'], uncertainty['mean_comp_ent_out'],
            xx, yy, store_dir, 'out_'+suffix)
    elif uncertainty == 'ensemble':
        uncertainty = estimate_uncertainty_ensembles(model_inp, 'all', dyna_model,
            ensemble_size, numb_samps = numb_samples)
        plot_hm_dep_uncertainty(uncertainty['total_ent'], uncertainty['mean_comp_ent'], 
            xx, yy, store_dir, suffix)
    elif uncertainty == 'nflows':
        uncertainty = estimate_uncertainty_nflows(model_inp, dyna_model,
            numb_samps = numb_samples)
        plot_total_ent_hm(uncertainty['total_ent_base'], 
            xx, yy, store_dir,'base_'+suffix)
        plot_total_ent_hm(uncertainty['total_ent_out'], 
            xx, yy, store_dir,'out_'+suffix)
    elif uncertainty == 'gp':
        #model_inp = model_inp.cpu().numpy()
        #mean_hat, std_hat = dyna_model.model.predict(model_inp, return_std=True)
        preds = dyna_model.model(model_inp)
        std_hat = preds.variance.sqrt().detach().cpu().mean(1)
        plot_total_ent_hm(std_hat, 
            xx, yy, store_dir, suffix)
