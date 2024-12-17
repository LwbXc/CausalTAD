import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pdb

class GMVSAE(nn.Module):
    def __init__(self, label_num, hidden_size, layer_num, n_cluster=10):
        super().__init__()
        self.pi_prior = nn.Parameter(torch.ones(n_cluster) / n_cluster)
        self.mu_prior = nn.Parameter(torch.zeros(n_cluster, hidden_size))
        self.log_var_prior = nn.Parameter(torch.randn(n_cluster, hidden_size))

        self.embedding = nn.Embedding(label_num, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, layer_num, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, layer_num, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc_out = nn.Linear(hidden_size, label_num)
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.n_cluster = n_cluster
        self.hidden_size = hidden_size

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def gaussian_pdf_log(self, x, mu, log_var):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / (torch.exp(log_var)), 1))

    def gaussian_pdfs_log(self, x, mus, log_vars):
        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c: c + 1, :], log_vars[c: c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def Loss(self, x_hat, targets, z_mu, z_sigma2_log, z, lengths):
        pi = self.pi_prior
        log_sigma2_c = self.log_var_prior
        mu_c = self.mu_prior

        reconstruction_loss = (self.crit(x_hat.transpose(1, 2), targets))
        reconstruction_loss = reconstruction_loss.sum(dim=-1)

        logits = -(torch.square(z.unsqueeze(1) - mu_c.unsqueeze(0)) / (2 * torch.exp(log_sigma2_c.unsqueeze(0)))).sum(-1)
        logits = F.softmax(logits, dim=-1) + 1e-10
        category_loss = torch.mean(torch.sum(logits * (torch.log(logits) - torch.log(pi).unsqueeze(0)), dim=-1))

        gaussian_loss = (self.gaussian_pdf_log(z, z_mu, z_sigma2_log).unsqueeze(1)
                         - self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)).mean()

        return reconstruction_loss,  category_loss + gaussian_loss / self.hidden_size

    def forward(self, src, trg, src_lengths, trg_lengths, train, c):
        batch_size = src.shape[0]
        e_input = self.embedding(src)
        d_input = self.embedding(trg)
        decoder_inputs = pack_padded_sequence(d_input, trg_lengths, batch_first=True, enforce_sorted=False)

        if train:
            encoder_inputs = pack_padded_sequence(e_input, src_lengths, batch_first=True, enforce_sorted=False)
            _, encoder_final_state = self.encoder(encoder_inputs)

            mu = self.fc_mu(encoder_final_state)
            logvar = self.fc_logvar(encoder_final_state)
            z = self.reparameterize(mu, logvar)
            decoder_outputs, _ = self.decoder(decoder_inputs, z)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
            output = self.fc_out(self.layer_norm(decoder_outputs))

            recon_loss, other_loss = self.Loss(output[:, :-1], trg[:, 1:], mu.squeeze(0), logvar.squeeze(0), z.squeeze(0), trg_lengths)
            return recon_loss.mean() + other_loss

        else:
            encoder_inputs = pack_padded_sequence(e_input, src_lengths, batch_first=True, enforce_sorted=False)
            _, encoder_final_state = self.encoder(encoder_inputs)

            mu = self.fc_mu(encoder_final_state)
            logvar = self.fc_logvar(encoder_final_state)
            z = self.reparameterize(mu, logvar)
            decoder_outputs, _ = self.decoder(decoder_inputs, z)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
            output = self.fc_out(self.layer_norm(decoder_outputs))

            recon_loss = self.crit(output[:, :-1].transpose(1, 2), trg[:, 1:])
            return recon_loss.sum(dim=-1)