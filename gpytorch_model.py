import os
import argparse
import time
import sys

import numpy as np
import torch
from base_model import base_model
import gpytorch
from gpytorch_utils import MultitaskGPModel
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood

class gpytorch_model(base_model):
    def __init__(self, input_preproc, output_preproc, 
            conditional_step = False, step_ahead_max = 30, device='cpu'):
        self.output_preproc = output_preproc
        self.input_preproc = input_preproc
        self.conditional_step = conditional_step
        self.step_ahead_max = step_ahead_max
        self.device = device
    
    def train_1d(self, epochs, data, blah):
        self.set_stats_1d(data)
        X = data[0].reshape(-1,1)
        y = data[1].reshape(-1,1)
        X = torch.tensor(X, dtype = torch.float32).to(self.device)
        y = torch.tensor(y, dtype = torch.float32).to(self.device)
        X = self.input_preproc(X, self.stats_inputs)
        y = self.output_preproc(y, self.stats_outputs)
        train_losses = []
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.shape[1])\
            .to(self.device)
        self.model = MultitaskGPModel(X, y, self.likelihood).to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1) 
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.mean().backward()
            optimizer.step()
            train_losses.append(loss.mean().item())
        return train_losses
    
    def loss_1d(self, data):
        X = data[0].reshape(-1,1)
        y = data[1].reshape(-1,1)
        X = torch.tensor(X, dtype = torch.float32).to(self.device)
        y = torch.tensor(y, dtype = torch.float32).to(self.device)
        X = self.input_preproc(X, self.stats_inputs)
        y = self.output_preproc(y, self.stats_outputs)
        self.model.eval()
        self.likelihood.eval()
        f_preds = self.model(X)
        #pred_post = self.model.likelihood(f_preds, X)
        pred_post = self.likelihood(f_preds)
        pred_post = torch.distributions.Normal(pred_post.mean, pred_post.stddev)
        loss = -pred_post.log_prob(y).mean().item()
        return loss

    def train(self, epochs, data_loader):
        train_losses = []
        self.set_stats(data_loader)
        batch = [tpl for tpl in data_loader.buffer]
        states, actions, reward, next_states, done, noisy_actions, index = map(np.stack, zip(*batch))
        states = torch.tensor(states, dtype = torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
        inps = torch.hstack([states, actions])
        outs = next_states
        X = self.input_preproc(inps, self.stats_inputs)
        y = self.output_preproc(outs, self.stats_outputs)
        train_losses = []
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.shape[1])\
            .to(self.device)
        self.model = MultitaskGPModel(X, y, self.likelihood).to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.mean().backward()
            optimizer.step()
            train_losses.append(loss.mean().item())
        return train_losses
    
    def loss(self, data_loader):
        with torch.no_grad():
            tot_inps = 0 
            loss = 0
            for data in data_loader:
                states = data[0]
                actions = data[1]
                next_states = data[3]
                states = torch.tensor(states, dtype = torch.float32).to(self.device)
                actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
                next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
                inps = torch.hstack([states, actions])
                outs = next_states
                X = self.input_preproc(inps, self.stats_inputs)
                y = self.output_preproc(outs, self.stats_outputs)
                self.model.eval()
                self.likelihood.eval()
                f_preds = self.model(X)
                #pred_post = self.model.likelihood(f_preds, X)
                pred_post = self.likelihood(f_preds)
                pred_post = torch.distributions.Normal(pred_post.mean, pred_post.stddev)
                loss += -pred_post.log_prob(y).sum().item()
                tot_inps += X.shape[0]
        return loss/tot_inps

    def detach_model(self):
        pass

    def attach_model(self):
        pass
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.save_constants(path)

    def sample(self, num_samps, context, kwargs ={}, 
        ensemble=False, ensemble_size=10):
        self.model.eval()
        self.likelihood.eval()
        if context.shape[0]>1000:
            means = []
            stds = []
            for i in range(int(np.ceil(context.shape[0])/1000)):
                f_preds = self.model(context[i*1000:(i+1)*1000])
                #y_preds = self.model.likelihood(f_preds, context)
                y_preds = self.likelihood(f_preds)
                mean = y_preds.mean
                std = y_preds.variance.sqrt()
                if len(y_preds.mean.shape) == 1:
                    mean = y_preds.mean.reshape(1,-1)
                    std = y_preds.variance.sqrt().reshape(1,-1)
                means.append(mean)
                stds.append(std)
            norm_rv = torch.distributions.normal.Normal(torch.vstack(means), torch.vstack(stds))
            samp = norm_rv.sample([num_samps])
            #samp = samp.reshape(samp.shape[2], samp.shape[0], samp.shape[1])
            samp = samp.reshape(samp.shape[1], samp.shape[0], samp.shape[2])

        else:
            f_preds = self.model(context)
            #y_preds = self.model.likelihood(f_preds, context)
            y_preds = self.likelihood(f_preds)
            mean = y_preds.mean
            std = y_preds.variance.sqrt()
            if len(y_preds.mean.shape) == 1:
                mean = y_preds.mean.reshape(1,-1)
                std = y_preds.variance.sqrt().reshape(1,-1)
            norm_rv = torch.distributions.normal.Normal(mean, std)
            samp = norm_rv.sample([num_samps])
            #samp = samp.reshape(samp.shape[2], samp.shape[0], samp.shape[1])
            samp = samp.reshape(samp.shape[1], samp.shape[0], samp.shape[2])
        return samp, 0, mean, std 

    def load_model(self, path):
        print('load model does not work')
