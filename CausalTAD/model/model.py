import pdb
import torch
import math
import torch.nn as nn
import numpy as np
from .vae import VAE
from .confidence import Confidence

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, layer_rnn, label_num) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.label_num = label_num
        self.confidence = Confidence(label_num, hidden_size)
        self.vae = VAE(input_size, hidden_size, layer_rnn, hidden_size, 0, label_num)
        self.road_embedding = nn.Embedding(self.label_num, hidden_size)
        self.projection_head = nn.Linear(hidden_size, label_num)
        self.sd_projection_head = nn.Linear(hidden_size, label_num)
        self.sd_loss = nn.NLLLoss(ignore_index=-1)
        self.log_soft = nn.LogSoftmax(dim=-1)

    def loss_fn(self, p_x, target):
        """
        Input:
        p_x (batch_size*seq_len, hidden_size): P(target|z)
        target (batch_size*seq_len) : the target sequences
        mask (batch_size*seq_len, vocab_size): the mask according to the road network
        """
        # masked softmax
        p_x = torch.exp(p_x)
        masked_sums = p_x.sum(dim=-1, keepdim=True) + 1e-6
        p_x = p_x/masked_sums
        p_x[:, self.label_num-1] = 1

        p_x = p_x[torch.arange(target.size(0)).to(target.device), target]
        nll = -torch.log(p_x)
        
        return nll

    def forward(self, src, trg, src_lengths=None, trg_lengths=None):
        """
        Input:
        src (batch_size, seq_len): the input sequence
        trg (batch_size, seq_len): the target sequence
        edge_list (2, edge_num): edges in the selected subgraph
        stage (int): indicates the first stage or the second stage
        src_length (batch_size): lengths of input sequences
        trg_length (batch_size): lengths of target sequences
        ---
        Output:
        loss (batch_size): loss
        """
        confidence, kl_confidence = self.confidence(src)

        batch_size, seq_len = src.size(0), src.size(1)+1
        cond_trg = src[torch.arange(batch_size), (src_lengths-1).long()].unsqueeze(1)
        cond_src = src[:, 0].unsqueeze(1)
        src = torch.cat((cond_src, cond_trg), dim=-1)
        sd = src
        src = self.road_embedding(src)
        src_lengths = torch.zeros([batch_size]).long() + 2

        label = torch.clone(trg[:, 1:])
        trg = self.road_embedding(trg)
        kl_loss, p_x, sd_p_x = self.vae.forward(src, trg, src_lengths, trg_lengths)


        batch_size, seq_len = label.size(0), label.size(1)
        label = label.view(-1)
        p_x = self.projection_head(p_x)
        p_x = p_x.view(batch_size*seq_len, -1)
        nll_loss = self.loss_fn(p_x, label)
        nll_loss = nll_loss.view(batch_size, seq_len)

        sd_p_x = self.sd_projection_head(sd_p_x)
        sd_p_x = sd_p_x.view(batch_size*2, -1)
        sd = sd.view(-1)
        sd_nll_loss = 0.1*self.loss_fn(sd_p_x, sd)
        sd_nll_loss = sd_nll_loss.view(batch_size*2, -1)

        mask = (torch.arange(seq_len-1).unsqueeze(0) < (trg_lengths.unsqueeze(1)-2)).to(kl_confidence.device)
        kl_confidence = mask*kl_confidence

        return nll_loss.sum(dim=-1), kl_loss, confidence.sum(dim=-1)+kl_confidence.sum(dim=-1), sd_nll_loss.mean()
