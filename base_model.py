import os
from joblib import dump, load

import numpy as np
import torch



class base_model(object):
    def __init__(self):
        pass

    def set_stats(self, data_loader):
        (state_mins, state_maxes, action_mins, action_maxes,
            state_means, state_stds, action_means, action_stds) = data_loader.get_stats()
        self.inp_mins = torch.tensor(np.hstack([state_mins, action_mins]),
                dtype = torch.float32).to(self.device)
        self.inp_maxes = torch.tensor(np.hstack([state_maxes, action_maxes]),
                dtype = torch.float32).to(self.device)
        self.inp_means = torch.tensor(np.hstack([state_means, action_means]),
                dtype = torch.float32).to(self.device)
        self.inp_stds = torch.tensor(np.hstack([state_stds, action_stds]),
                dtype = torch.float32).to(self.device)
        self.state_maxes = torch.tensor(state_maxes, dtype = torch.float32).to(self.device)
        self.state_mins = torch.tensor(state_mins, dtype = torch.float32).to(self.device)
        self.action_maxes = torch.tensor(action_maxes, dtype = torch.float32).to(self.device)
        self.action_mins = torch.tensor(action_mins, dtype = torch.float32).to(self.device)
        self.state_means = torch.tensor(state_means, dtype = torch.float32).to(self.device)
        self.state_stds = torch.tensor(state_stds, dtype = torch.float32).to(self.device)
        self.action_means = torch.tensor(action_means, dtype = torch.float32).to(self.device)
        self.action_stds = torch.tensor(action_stds, dtype = torch.float32).to(self.device)
        self.stats_inputs = [self.inp_mins, self.inp_maxes,
            self.inp_means, self.inp_stds]
        self.stats_outputs = [self.state_mins, self.state_maxes,
            self.state_means, self.state_stds]

    def set_stats_1d(self, data):
        self.stats_inputs = [data[0].min(), data[0].max(), data[0].mean(), data[0].std()]
        self.stats_outputs = [data[1].min(), data[1].max(), data[1].mean(), data[1].std()]

    def save_constants(self, path):
        model_dir = os.path.dirname(path)
        inp_stats_path = os.path.join(model_dir, 'input_stats.txt')
        out_stats_path = os.path.join(model_dir, 'output_stats.txt')
        dump(self.stats_inputs, inp_stats_path)
        dump(self.stats_outputs, out_stats_path)

    def load_constants(self, path):
        model_dir = os.path.dirname(path)
        inp_stats_path = os.path.join(model_dir, 'input_stats.txt')
        out_stats_path = os.path.join(model_dir, 'output_stats.txt')
        self.stats_inputs = load(inp_stats_path)
        self.stats_outputs = load(out_stats_path)
        
