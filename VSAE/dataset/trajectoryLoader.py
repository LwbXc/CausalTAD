from copy import deepcopy
import os
import json
import pickle
import pdb
from random import random, shuffle
import torch

class TrajectoryLoader:

    def __init__(self, trajectory_path: str, batch_size: int, label_num: int, valid: bool) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.label_num = label_num
        self.load_data(trajectory_path, valid)
        self.bos_eos_pad()
        self.batch_preprocess()

    def shuffle_dataloader(self):
        shuffle(self.data)

    def load_data(self, trajectory_path, valid):
        dataset = pickle.load(open(trajectory_path, 'rb'))
        self.data = []
        if valid:
            dataset = list(dataset.values())
            for line in dataset[:50]:
                traj = line['grid']
                self.data.append(traj)
        else:
            for line in dataset.values():
                traj = line['grid']
                self.data.append(traj)
        # shuffle(self.data)

    def bos_eos_pad(self):
        self.bos = self.label_num-3
        self.eos = self.label_num-2
        self.pad = self.label_num-1

    def batch_preprocess(self):

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
            for item in cur_batch:
                src_length.append(len(item))
                trg_batch.append([self.bos] + deepcopy(item) + [self.eos])
                trg_length.append(len(trg_batch[-1]))
            max_length = max(src_length)

            for item in cur_batch:
                item += [self.pad]*(max_length-len(item))
            
            for item in trg_batch:
                item += [self.pad]*(max_length+2-len(item))
            
            self.src_data_batchs.append(torch.LongTensor(cur_batch))
            self.trg_data_batchs.append(torch.LongTensor(trg_batch))
            self.src_length_batchs.append(torch.IntTensor(src_length))
            self.trg_length_batchs.append(torch.IntTensor(trg_length))