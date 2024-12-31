from copy import deepcopy
import os
import json
import pickle
import pdb
from random import random, shuffle
import torch

class TrajectoryLoader:

    def __init__(self, trajectory_path: str, batch_size: int, label_num: int, rho: float = 1) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.label_num = label_num
        self.load_data(trajectory_path)
        self.bos_eos_pad()
        self.batch_preprocess(rho)

    def shuffle_dataloader(self):
        shuffle(self.data)

    def load_data(self, trajectory_path):
        dataset = pickle.load(open(trajectory_path, 'rb'))
        self.data = []
        for line in dataset.values():
            traj = line['grid']
            self.data.append(traj)

    def bos_eos_pad(self):
        self.bos = self.label_num-3
        self.eos = self.label_num-2
        self.pad = self.label_num-1

    def batch_preprocess(self, rho):

        self.src_data_batchs = []
        self.trg_data_batchs = []
        self.src_length_batchs = []
        self.trg_length_batchs = []

        for i in range(0, len(self.data), self.batch_size):
            if i+self.batch_size>len(self.data):
                cur_batch = self.data[i:len(self.data)]
            else:
                cur_batch = self.data[i:i+self.batch_size]
            
            src_length = []
            trg_length = []
            trg_batch = []
            src_batch = []
            for item in cur_batch:
                src_length.append(2)
                trg_batch.append([self.bos] + deepcopy(item[:int(rho*len(item))]) + [self.eos])
                src_batch.append([item[0], item[-1]])
                trg_length.append(len(trg_batch[-1]))
            max_length = max(trg_length)
            
            for item in trg_batch:
                item += [self.pad]*(max_length-len(item))
            
            self.src_data_batchs.append(torch.LongTensor(src_batch))
            self.trg_data_batchs.append(torch.LongTensor(trg_batch))
            self.src_length_batchs.append(torch.IntTensor(src_length))
            self.trg_length_batchs.append(torch.IntTensor(trg_length))