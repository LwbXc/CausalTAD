import pdb
import torch
import torch.nn as nn

class RoadEmbedding(nn.Module):

    def __init__(self, embed_size, hidden_size) -> None:
        super().__init__()
        self.embedding = nn.Embedding(embed_size, hidden_size, -1)
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.embedding(x)