import pdb
from random import random

import torch
import torch.nn as nn

from .encoder import EncoderRNN
from .decoder import DecoderRNN

class VAE(nn.Module):
    
    def __init__(self, input_size, hidden_size, layer_num, latent_num, dropout, label_num) -> None:
        super().__init__()
        self.enc = EncoderRNN(input_size, hidden_size, layer_num, latent_num, dropout, True)
        self.dec = DecoderRNN(input_size, hidden_size, layer_num, latent_num, dropout, label_num)
        self.nll = nn.NLLLoss(ignore_index=label_num-1, reduction='none')
        self.predict = nn.Linear(hidden_size, label_num)
        self.label_num = label_num

    def loss_fn(self, p_x, target):
        """
        Input:
        p_x (batch_size*seq_len, hidden_size): P(target|z)
        target (batch_size*seq_len) : the target sequences
        mask (batch_size*seq_len, vocab_size): the mask according to the road network
        """
        # masked softmax
        p_x = torch.exp(self.predict(p_x))
        masked_sums = p_x.sum(dim=-1, keepdim=True) + 1e-6
        p_x = p_x/masked_sums
        p_x[:, self.label_num-1] = 1

        p_x = p_x[torch.arange(target.size(0)).to(target.device), target]
        nll = -torch.log(p_x)
        # (batch_size, latent_num)
        return nll

    def forward(self, src, trg, label, src_lengths=None, trg_lengths=None):
        """
        Input:
        src (batch_size, seq_len, hidden_size): input sequence tensor
        trg (batch_size, seq_len, hidden_size): the target sequence tensor
        label (batch_size, seq_len): the target sequence label
        edge_list (2, edge_num): edges in the selected subgraph
        src_length (batch_size): lengths of input sequences
        trg_length (batch_size): lengths of target sequences
        ---
        Output:
        output (seq_len, batch_size, hidden_size)
        hidden_state (layer_num, batch_size, hidden_size)
        """
        z = self.enc.forward(src, src_lengths)
        
        # (batch_size, seq_len, hidden_size)
        p_x = self.dec.forward(z, trg[:, :-1], trg_lengths-1)
        batch_size, seq_len = label.size(0), label.size(1)
        
        # (batch_size, seq_len) => (batch_size*seq_len, )
        label = label.view(-1)
        p_x = p_x.view(batch_size*seq_len, -1)

        nll_loss = self.loss_fn(p_x, label)
        nll_loss = nll_loss.view(batch_size, seq_len)
        
        return nll_loss