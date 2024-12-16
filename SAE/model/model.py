import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import VAE
from .embedding import RoadEmbedding

class Model(nn.Module):

    def __init__(self, label_num, hidden_size, input_size, layer_num, latent_num, dropout, offline=True) -> None:
        super().__init__()
        # self.road_embeddings = PretrainEmbedding(label_start, hidden_size)
        self.road_embedding = RoadEmbedding(label_num, hidden_size)
        self.vae = VAE(input_size, hidden_size, layer_num, latent_num, dropout, label_num)
        self.dropout = nn.Dropout(dropout)
        self.offline = offline
    
    def forward(self, src, trg, src_lengths, trg_lengths):
        """
        Input:
        src (batch_size, src_seq_len): input source sequences
        trg (batch_size, trg_seq_len): input target sequences
        src_lengths (batch_size): lengths of input source sequences
        trg_lengths (batch_size): lengths of input target sequences
        edge_list (2, edge_num): edges in the selected subgraph
        ---
        Output:
        nll_loss (batch_size, seq_length): negative log likehihood
        kl_loss (batch_size): KL-divergence
        """
        if self.offline==True:
            src = self.road_embedding(src)
            src = self.dropout(src)
        else:
            batch_size = src.size(0)
            cond_trg = src[torch.arange(batch_size), (src_lengths-1).long()].unsqueeze(1)
            cond_src = src[:, 0].unsqueeze(1)
            src = torch.cat((cond_src, cond_trg), dim=-1)
            src = self.road_embedding(src)
            src_lengths = torch.zeros([batch_size]).long() + 2

        label = torch.clone(trg[:, 1:])
        trg = self.road_embedding(trg)
        nll_loss = self.vae.forward(src, trg, label, src_lengths, trg_lengths)
        return nll_loss