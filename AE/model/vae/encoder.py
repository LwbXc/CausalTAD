import pdb
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, layer_num, latent_size, dropout, bidirectional=True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size%self.num_directions == 0
        self.lstm = nn.GRU(input_size, hidden_size//self.num_directions, layer_num, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.projection = nn.Linear(layer_num*hidden_size, latent_size)
        self.dropout = nn.Dropout(0.15)

    def forward(self, input, lengths=None):
        """
        Input:
        input (batch_size, seq_len, hidden_size): padded input sequence tensor
        lengths (batch_size): lengths of input sequences
        ---
        Output:
        q_z: A normal distribution
        """
        if lengths is not None:
            packed_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        # hidden/cell: (layer_num*2, batch_size, hidden_size/2)
        output, hidden = self.lstm(packed_input)

        # if lengths is not None:
        #     output = pad_packed_sequence(output)[0]
        
        if self.num_directions==2:
            batch_size, half_hidden = hidden.size(1), hidden.size(2)
            # (batch_size, seq_len*hidden_size)
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        hidden = self.projection(hidden)
        hidden = self.dropout(hidden)
        return hidden