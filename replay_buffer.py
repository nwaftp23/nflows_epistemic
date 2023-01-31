import os
import random
import pickle

import numpy as np


class ReplayMemory:
    def __init__(self, capacity, batch_size, shuffle = True, bootstrap = False,
            ensemble_size = 5, action_seq = False, conditional_step = False,
            step_ahead_max = 30):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bootstrap = bootstrap
        self.ensemble_size = ensemble_size
        self.action_seq = action_seq
        self.conditional_step = conditional_step
        self.step_ahead_max = step_ahead_max

    def get_num_episodes(self, reduce_buf = False):
        state, action, reward, next_state, done, noisy_actions, index = \
            map(np.stack, zip(*self.buffer))
        self.num_episodes = done.sum()
        if not reduce_buf:
            self.episode_length = np.where(done)[0][0]+1
        self.numpy_buffer = np.array(self.buffer)
    
    def add_index(self):
        self.buffer = [val+(i,) for i, val in enumerate(self.buffer)]

    def add_to_buffer(self, addition):
        self.buffer += addition

    def remove_portion(self, rep_buffer):
        indeces = [i[-1] for i in rep_buffer]
        self.buffer = [i for i in self.buffer if i[-1] not in indeces]

    def dyna_buffer(self, nstep):
        state, action, reward, next_state, done, noisy_actions, index = \
            map(np.stack, zip(*self.buffer))
        (split_state, split_next_state, split_action, 
            split_done, split_reward, split_noisy_actions, split_indeces) = \
            self.episode_split(state, action, reward, next_state, done, noisy_actions)
        # nstep vs nstep-1
        split_state = [epi[:-(nstep-1)] for epi in split_state 
                if epi[:-(nstep-1)].shape[0] != 0]
        split_next_state = [epi[(nstep-1):] for epi in split_next_state 
                if epi[:-(nstep-1)].shape[0] != 0]
        if self.action_seq:
            split_action = [self.make_action_seqs(epi, nstep) for epi in split_action 
                    if epi[:-(nstep-1)].shape[0] != 0]
        else:
            split_action = [epi[:-(nstep-1)] for epi in split_action 
                    if epi[:-(nstep-1)].shape[0] != 0]
        split_reward = [epi[:-(nstep-1)] for epi in split_reward 
                if epi[:-(nstep-1)].shape[0] != 0]
        split_done = [epi[:-(nstep-1)] for epi in split_done 
                if epi[:-(nstep-1)].shape[0] != 0]
        split_noisy_actions = [epi[:-(nstep-1)] for epi in split_noisy_actions 
                if epi[:-(nstep-1)].shape[0] != 0]
        split_indeces = [epi[:-(nstep-1)] for epi in split_indeces 
                if epi[:-(nstep-1)].shape[0] != 0]
        nstep_states = np.vstack(split_state)
        nstep_next_states = np.vstack(split_next_state)
        nstep_action = np.vstack(split_action)
        nstep_reward = np.hstack(split_reward)
        nstep_done = np.hstack(split_done)
        nstep_noisy_actions = np.vstack(split_noisy_actions)
        nstep_indeces = np.vstack(split_indeces)
        replay_buffer = [(nstep_states[i], nstep_action[i], nstep_reward[i], 
            nstep_next_states[i], nstep_done[i], nstep_noisy_actions[i])
            for i in range(nstep_states.shape[0])]
        return replay_buffer

    def episode_split(self, state, action, reward, next_state, done, noisy_actions, indeces):
        split_state = np.split(state,np.where(done)[0]+1)
        split_next_state = np.split(next_state,np.where(done)[0]+1)
        split_action = np.split(action, np.where(done)[0]+1)
        split_done = np.split(done,np.where(done)[0]+1)
        split_reward = np.split(reward,np.where(done)[0]+1)
        split_noisy_actions = np.split(noisy_actions,np.where(done)[0]+1)
        split_indeces = np.split(noisy_actions,np.where(done)[0]+1)
        return (split_state, split_next_state, split_action, split_done,
                split_reward, split_noisy_actions, split_indeces)

    def make_action_seqs(self, actions, nstep):
        seqs = []
        for i in range((actions.shape[0]-nstep+1)):
            seqs.append(actions[i:(i+nstep)].ravel())
        seqs = np.vstack(seqs)
        return seqs

    def push(self, state, action, reward, next_state, done, noisy_action):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, noisy_action)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, noisy_action, index = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, noisy_action, index
    
    def sample_step(self, batch_size):
        # replace 10 with batch_size
        episodes = np.random.choice(self.num_episodes, size=10, replace=False)
        step_ahead = np.random.randint(self.step_ahead_max)+1
        step_start = np.random.randint(self.episode_length, size=10)
        start_idxs = (episodes-1)*self.episode_length+step_start
        end_idxs = start_idxs+step_ahead
        state, action, reward, _, done, noisy_action, index = map(np.stack, 
            zip(*self.numpy_buffer[start_idxs]))
        next_state, _, _, _, _, _, _ = map(np.stack, zip(*self.numpy_buffer[end_idxs]))
        return state, action, reward, next_state, done, noisy_action, step_ahead
    
    def sample_ordered(self, episode_length):
        state, action, reward, next_state, done, noisy_action, index = map(np.stack, 
                                                      zip(*self.buffer[:(episode_length*2)]))
        return state, action, reward, next_state, done, noisy_action
    
    def shuffle_indices(self, batch_size):
        import pdb; pdb.set_trace()
        order = np.random.choice(len(self.buffer), len(self.buffer), replace=False)
        total_points = len(self.buffer)
        batches = int(np.ceil(total_points/batch_size))
        self.batch_indxs = np.array_split(order, batches)

    def get_batch(self, indxs):
        batch = [self.buffer[i] for i in indxs]
        state, action, reward, next_state, done, noisy_actions, index = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, noisy_actions
    
    def get_batch_indices(self, shuffle=True):
        data_points = len(self.buffer)
        num_chunks = int(np.ceil(data_points/self.batch_size))
        if shuffle:
            indices = np.random.permutation(data_points)
        else:
            indices = np.arange(data_points)
        self.split_indices = np.array_split(indices, num_chunks)
    
    def reduce_buffer(self, size):
        #batch = random.sample(self.buffer, size)
        batch = self.buffer[:size]
        self.buffer = batch
        self.get_num_episodes(reduce_buf=True)
    

    def get_stats(self):
        state, action, reward, next_state, done, _, _ = map(np.stack, zip(*self.buffer))
        state_mins = state.min(0)
        state_maxes = state.max(0)
        action_mins = action.min(0)
        action_maxes = action.max(0)
        state_means = state.mean(0)
        state_stds = state.std(0)
        action_means = action.mean(0)
        action_stds = action.std(0)
        return (state_mins, state_maxes, action_mins, action_maxes, 
            state_means, state_stds, action_means, action_stds)
                
    def __len__(self):
        return len(self.buffer)
    
    def __iter__(self):
        self.get_batch_indices(shuffle = self.shuffle)
        self.counter = 0
        self.num_updates = (len(self.buffer) // self.batch_size + 1)
        return self

    def __next__(self):
        if not self.bootstrap:
            if not self.conditional_step:
                try:
                    batch = self.get_batch(self.split_indices[self.counter])
                except IndexError:
                    raise StopIteration
            else:
                if self.counter > self.num_updates:
                    raise StopIteration
                batch = self.sample_step(self.batch_size)
        else:
            if not self.conditional_step:
                if self.counter > self.num_updates:
                    raise StopIteration
                else:
                    batches = []
                    for i in range(self.ensemble_size):
                        batch = self.sample(self.batch_size)
                        batches.append(batch)
                states = [i[0] for i in batches]
                actions = [i[1] for i in batches]
                rewards = [i[2] for i in batches]
                next_states = [i[3] for i in batches]
                dones = [i[4] for i in batches]
                noisy_actions = [i[5] for i in batches]
                batch = [np.stack(states), np.stack(actions), np.stack(rewards), 
                        np.stack(next_states), np.stack(dones), np.stack(noisy_actions)]
            else:
                if self.counter > self.num_updates:
                    raise StopIteration
                batch = self.sample_step(self.batch_size) 
        self.counter += 1
        return batch

def load_mem(args, memory, store_dir, test=False):
    if args.rc_data:
        if not test:
            filename = 'train_rc_car_data.pkl'
        else:
            filename = 'test_rc_car_data.pkl'
    else:
        if not test:
            filename = 'buffer_seed'+str(args.noise_seed)+'.pkl'
        else:
            filename = 'buffertest_seed'+str(args.noise_seed)+'.pkl'
    filepath = os.path.join(store_dir, filename)
    file = open(filepath, 'rb')
    data = pickle.load(file)
    file.close()
    memory.buffer = data
    memory.add_index()
    memory.get_num_episodes()
    if args.nstep > 1:
        rep_buffer = memory.dyna_buffer(args.nstep)
        memory.buffer = rep_buffer

def load_mem_uncertain(args, memory, store_dir, test=False, oracle=False):
    if not test:
        filename = 'PureRand_buffertest_seed'+str(args.noise_seed)+'.pkl'
    else:
        filename = 'SAC_buffertest_seed'+str(args.noise_seed)+'.pkl'
    if oracle:
        filename = 'PureRand_buffer_seed'+str(args.noise_seed)+'.pkl'
    branch_folder = 'noiseweight'+str(args.noise_weight)+'_'
    if args.fat_tail:
        branch_folder = branch_folder+'fattail'
    elif args.valley_distribution:
        branch_folder = branch_folder+'valleydistribution'
    else:
        branch_folder = branch_folder+'modes'+str(args.modes)
    buffer_dir = os.path.join(store_dir, branch_folder)
    filepath = os.path.join(buffer_dir, filename)
    file = open(filepath, 'rb')
    data = pickle.load(file)
    file.close()
    memory.buffer = data
    memory.add_index()
    memory.get_num_episodes()
    if args.nstep > 1:
        rep_buffer = memory.dyna_buffer(args.nstep)
        memory.buffer = rep_buffer
    return buffer_dir


def load_mem_basic(memory, path):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    memory.buffer = data
    memory.add_index()
    memory.get_num_episodes()

