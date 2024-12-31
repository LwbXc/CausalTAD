import numpy as np
import torch
import pdb
import torch.nn as nn

from .vae import GMVSAE

class Model(nn.Module):
    def __init__(self, label_num, hidden_size, layer_num, n_cluster=10):
        super().__init__()
        self.model = GMVSAE(label_num, hidden_size, layer_num, n_cluster=n_cluster)

        self.crit = nn.CrossEntropyLoss(reduction='none')
        self.detect = nn.CrossEntropyLoss(reduction='none')
        self.n_cluster = n_cluster

    def forward(self, src, trg, src_lengths, trg_lengths, train=True):
        if train:
            loss = self.model(src, trg, src_lengths, trg_lengths, train=train, c=-1)
            return loss
        else:
            loss = self.model(src, trg, src_lengths, trg_lengths, train=train, c=-1)
            return loss