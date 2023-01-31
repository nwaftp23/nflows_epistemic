import os
import argparse
import time
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from base_model import base_model
from mc_drop_utils import build_mc_drop

class mc_drop_model(base_model):
    def __init__(self, num_layers, hids, dims, context_dims, 
            lr, gamma, device, input_preproc, output_preproc, 
            conditional_step = False, step_ahead_max = 30):
        self.dims = dims
        self.model = build_mc_drop(num_layers=num_layers, hids=hids, dims=dims, 
            context_dims=context_dims, device = device).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size = 1, gamma = gamma)
        self.device = device
        # if normalize convert to beta maybe?
        self.output_preproc = output_preproc
        self.input_preproc = input_preproc
        self.conditional_step = conditional_step
        self.step_ahead_max = step_ahead_max
    
    def train_1d(self, epochs, data, un_normalize):
        train_losses = []
        self.set_stats_1d(data)
        for epoch in range(epochs):
            running_train_loss = 0
            inp = data[0].reshape(-1,1)
            out = data[1].reshape(-1,1)
            inps = torch.tensor(inp, dtype = torch.float32).to(self.device)
            outs = torch.tensor(out, dtype = torch.float32).to(self.device)
            inps = self.input_preproc(inps, self.stats_inputs)
            outs = self.output_preproc(outs, self.stats_outputs)
            self.optimizer.zero_grad()
            mu, sig, mask_prob = self.model(inps)
            loss = -self.model.loss_val(mu, sig, outs)
            loss.backward()
            if loss.isnan():
                print('loss is nan')
            self.optimizer.step()
            train_loss = loss.cpu().detach()
            train_losses.append(train_loss)
            if (epoch + 1) % 1000 == 0:
                print(f'Epoch {epoch+1} loss: {train_loss.item()}')
        return train_losses
    
    def loss_1d(self, data):
        with torch.no_grad():
            inp = data[0].reshape(-1,1)
            out = data[1].reshape(-1,1)
            inps = torch.tensor(inp, dtype = torch.float32).to(self.device)
            outs = torch.tensor(out, dtype = torch.float32).to(self.device)
            inps = self.input_preproc(inps, self.stats_inputs)
            outs = self.output_preproc(outs, self.stats_outputs)
            loss = 0
            mask_samples = 10
            for ei in range(mask_samples):
                mu, sig, mask_prob = self.model(inps)
                comp_loss = self.model.log_prob(mu, sig, outs)
                loss += (torch.exp(comp_loss)*1/mask_samples)
            loss = torch.log(loss)
            loss[loss.isinf()] = loss[~loss.isinf()].min()
            loss = -loss.mean().cpu().detach()
        return loss 

    def train(self, epochs, data_loader):
        train_losses = []
        self.set_stats(data_loader)
        for epoch in range(epochs):
            running_train_loss = 0
            total_inputs = 0
            for data in data_loader:
                states = data[0]
                actions = data[1]
                next_states = data[3]
                states = torch.tensor(states, dtype = torch.float32).to(self.device)
                actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
                next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
                inps = torch.hstack([states, actions])
                outs = next_states
                inp_stats = self.stats_inputs + [states]
                out_stats = self.stats_outputs + [states]
                inps = self.input_preproc(inps, inp_stats)
                outs = self.output_preproc(outs, out_stats)
                self.optimizer.zero_grad()
                # set random mask each time??
                mu, sig, mask_prob = self.model(inps)
                loss = -self.model.loss_val(mu, sig, outs)
                loss.backward()
                if loss.isnan():
                    print('loss is nan')
                self.optimizer.step()
                running_train_loss += loss.cpu().detach()*states.shape[0]
                total_inputs += states.shape[0]
            running_train_loss = running_train_loss/total_inputs
            train_losses.append(running_train_loss)
        return train_losses
    
    def loss(self, data_loader):
        running_loss = 0
        total_inputs = 0
        with torch.no_grad():
            for data in data_loader:
                states = data[0]
                actions = data[1]
                next_states = data[3]
                states = torch.tensor(states, dtype = torch.float32).to(self.device)
                actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
                next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
                inps = torch.hstack([states, actions])
                outs = next_states
                inp_stats = self.stats_inputs + [states]
                out_stats = self.stats_outputs + [states]
                inps = self.input_preproc(inps, inp_stats)
                outs = self.output_preproc(outs, out_stats)
                loss = 0
                mask_samples = 10
                for ei in range(mask_samples):
                    mu, sig, mask_prob = self.model(inps)
                    comp_loss = self.model.log_prob(mu, sig, outs)
                    loss += (torch.exp(comp_loss)*1/mask_samples)
                loss = torch.log(loss)
                loss[loss.isinf()] = loss[~loss.isinf()].min()
                running_loss += -loss.sum().cpu().detach() 
                total_inputs += states.shape[0]
        running_loss = running_loss/total_inputs
        return running_loss 

    def detach_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def attach_model(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.save_constants(path)
    
    def load_model(self, path):
        import pdb; pdb.set_trace()

    def sample(self, numb_samps, context, kwargs={},
            ensemble = True, ensemble_size = 10):
        samp, mask_prob, mu, sig = self.model.sample(numb_samps, context=context, kwargs=kwargs)
        return samp, mask_prob, mu, sig
    
    def sample_and_log_prob(self, numb_samps, context, kwargs={},
             ensemble = True, ensemble_size = 10):
        samp, log_prob, mask_prob, mu, sig = self.model.sample_and_log_prob(numb_samps, 
            context=context, kwargs=kwargs)
        return samp, log_prob, mask_prob, mu, sig
