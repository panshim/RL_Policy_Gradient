# This is a newly implemented script, with reference to:
# https://github.com/aravindr93/mjrl/blob/master/mjrl/utils/fc_network.py
# It is an implementation of LSTM module used for the policy. It involves
# customized implementation of prediction with batch data ("forward" method)
# used for efficient gradient calculation, and prediction with sequential
# data ("run_sequence" method) used for sampling in the simulation.

import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class RNNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='relu',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(RNNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        # hidden layers
        # self.batch_norm = nn.BatchNorm1d(self.layer_sizes[0])
        self.fc_layer = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.lstm = nn.LSTM(self.layer_sizes[1], self.layer_sizes[2], 1, batch_first=True)
        self.output_layer = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, obs):
        # TODO(Aravind): Remove clamping to CPU
        # This is a temp change that should be fixed shortly
        x = pad_sequence(obs, batch_first=True)
        lengths = [len(o) for o in obs]
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        out = (out - self.in_shift)/(self.in_scale + 1e-8)

        batch_size = out.shape[0]
        max_length = out.shape[1]
        out = out.view(batch_size * max_length, -1)
        #out = self.batch_norm(out)
        out = self.nonlinearity(self.fc_layer(out))
        out = out.view(batch_size, max_length, -1)
        # out = pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        # out = self.lstm(out)[0].data
        out = self.lstm(out)[0]
        out_r = []
        for i, l in enumerate(lengths):
            out_r.append(out[i, 0:l, :])
        out_r = torch.cat(out_r)
        out_r = self.nonlinearity(out_r)
        out_r = self.output_layer(out_r)
        out_r = out_r * self.out_scale + self.out_shift
        return out_r

    def run_sequence(self, x, h=None, c=None):
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        out = (out - self.in_shift)/(self.in_scale + 1e-8)
        #print(self.batch_norm.running_mean.shape)
        out = self.nonlinearity(self.fc_layer(out))
        out = out.view(1, 1, -1)
        if (h is None) and (c is None):
            out, (h_next, c_next) = self.lstm(out)
        else:
            out, (h_next, c_next) = self.lstm(out, (h, c))
        out = self.nonlinearity(out)
        out = out.view(1, -1)
        # print(out.shape)
        out = self.output_layer(out)
        out = out * self.out_scale + self.out_shift
        return out, h_next, c_next
