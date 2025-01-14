import os
import json
from turtle import position
import torch
import pdb
import torch.nn as nn
import time
import numpy as np

from torch.optim import SGD, Adam
from torch.nn.utils import clip_grad_value_

from dataset import TrajectoryLoader
from model import Model
from params import Params
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class Trainer:

    def __init__(self, config, save_path, cuda_devices=[1], load_model=None) -> None:
        self.config = config
        self.save_path = save_path
        self.load_model = load_model
        self.cuda_devices = cuda_devices
        self.device = 'cuda:0'
        self.params = Params()
        self.label_num = self.config['grid_size'][0]*self.config['grid_size'][1] + 3

        self.model = Model(self.label_num, self.params.hidden_size, self.params.hidden_size, self.params.layer_num, self.params.latent_num, self.params.dropout)
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.label_num-1)
        if torch.cuda.device_count()>1 and len(cuda_devices)>1:
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        self.model = self.model.to(self.device)
        if load_model!=None:
            checkpoint = torch.load(os.path.join(self.load_model))
            self.model.load_state_dict(checkpoint['model'])

        self.optimizer = Adam([{"params":self.model.parameters(
                ), 'lr':self.params.lr}], weight_decay=self.params.weight_decay)

    def train(self, epochs: int, train_dataloader: TrajectoryLoader, valid_dataloader: dict):
        
        best_performance = float('-inf')
        for epoch in range(epochs):
            avg_loss = 0

            print(self.params.batch_size, len(train_dataloader.src_data_batchs))
            train_dataloader.shuffle_dataloader()
            self.model.train()
            for i, data in enumerate(train_dataloader.src_data_batchs):
                src, trg, src_lengths, trg_lengths = data.to(self.device), train_dataloader.trg_data_batchs[i].to(self.device), train_dataloader.src_length_batchs[i], train_dataloader.trg_length_batchs[i]
                nll_loss = self.model.forward(src, trg, src_lengths, trg_lengths)
                nll_loss = nll_loss.sum(dim=-1)
                loss = nll_loss.mean()
                avg_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                post = "Train epoch:{}, iter:{}, avgloss:{:.4f}, loss:{:.4f}".format(epoch, i, avg_loss/(i+1), loss.item())

                if i%10==0:
                    print(post)

            valid_prob = {0:np.array([]), 1:np.array([])}
            self.model.eval()
            with torch.no_grad():
                for k,dataloader in valid_dataloader.items():
                    for i, data in enumerate(dataloader.src_data_batchs):
                        src, trg, src_lengths, trg_lengths = data.to(self.device), dataloader.trg_data_batchs[i].to(self.device), dataloader.src_length_batchs[i], dataloader.trg_length_batchs[i]
                        nll_loss = self.model.forward(src, trg, src_lengths, trg_lengths)
                        nll_loss = nll_loss.sum(dim=-1)
                        nll_loss = nll_loss.cpu().detach().numpy()
                        valid_prob[k] = np.concatenate((valid_prob[k], nll_loss))
            
            score = np.concatenate((valid_prob[1], valid_prob[0]))
            label = np.concatenate((np.ones(len(valid_prob[1])), np.zeros(len(valid_prob[0]))))
            pre, rec, _t = precision_recall_curve(label, score)
            area = auc(rec, pre)


            if best_performance<area:
                best_performance = area
                best_epoch = epoch
                self.save()

            valid_post = f"Best performance {best_performance}, best epoch {best_epoch}, cur performance {area}, cur epoch {epoch}"
            print(valid_post)

            if epoch-best_epoch>=5:
                break

        with open("log.txt", 'a') as f:
            f.write(post + f', best epoch:{best_epoch}\n')
        
    def test(self, normal_dataloader, abnormal_dataloader):
        
        normal_prob = np.array([])
        abnormal_prob = np.array([])

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(normal_dataloader.src_data_batchs):
                src, trg, src_lengths, trg_lengths = data.to(self.device), normal_dataloader.trg_data_batchs[i].to(self.device), normal_dataloader.src_length_batchs[i], normal_dataloader.trg_length_batchs[i]
                nll_loss = self.model.forward(src, trg, src_lengths, trg_lengths)
                nll_loss = nll_loss.sum(dim=-1)
                nll_loss = nll_loss.cpu().detach().numpy()
                normal_prob = np.concatenate((normal_prob, nll_loss))

            for i, data in enumerate(abnormal_dataloader.src_data_batchs):
                src, trg, src_lengths, trg_lengths = data.to(self.device), abnormal_dataloader.trg_data_batchs[i].to(self.device), abnormal_dataloader.src_length_batchs[i], abnormal_dataloader.trg_length_batchs[i]
                nll_loss = self.model.forward(src, trg, src_lengths, trg_lengths)
                nll_loss = nll_loss.sum(dim=-1)
                nll_loss = nll_loss.cpu().detach().numpy()
                abnormal_prob = np.concatenate((abnormal_prob, nll_loss))
        
        score = np.concatenate((abnormal_prob, normal_prob))
        label = np.concatenate((np.ones(len(abnormal_prob)), np.zeros(len(normal_prob))))
        pre, rec, _t = precision_recall_curve(label, score)
        pr = auc(rec, pre)
        roc = roc_auc_score(label, score)

        return round(pr, 4)

    def save(self):
        if torch.cuda.device_count()>1 and len(self.cuda_devices)>1:
            state = {
                'model': self.model.module.state_dict()
            }
        else:
            state = {
                'model': self.model.state_dict()
            }
        torch.save(state, os.path.join(self.save_path))



if __name__=="__main__":
    if not os.path.exists("./save"):
        os.mkdir("./save")

    for city in ["xian", "porto"]:
        root_path = f"../datasets/{city}"
        batch_size = 128
        runs = 5
        config = json.load(open(os.path.join(root_path, "config.json")))


        pr_auc_list = dict()
        head = dict()
        for _i in range(runs):
            trainer = Trainer(config, save_path=f'save/{city}_{_i}.pth', load_model=None)

            train_dataloader = TrajectoryLoader(os.path.join(root_path, "train-grid.pkl"), batch_size, label_num=config['grid_size'][0]*config['grid_size'][1]+3)
            valid_dataloader = {0: TrajectoryLoader(os.path.join(root_path, "valid-normal-grid.pkl"), batch_size, label_num=config['grid_size'][0]*config['grid_size'][1]+3),
                                1: TrajectoryLoader(os.path.join(root_path, "valid-abnormal-grid.pkl"), batch_size, label_num=config['grid_size'][0]*config['grid_size'][1]+3)}
            trainer.train(200, train_dataloader, valid_dataloader)


            alpha = [0.1, 0.2, 0.3]
            distance = [1, 2, 3]
            ratio = [0.5, 0.7, 1.0]
            trainer = Trainer(config, save_path=None, load_model=f'save/{city}_{_i}.pth')

            for r in ratio:
                normal_dataloader = TrajectoryLoader(os.path.join(root_path, "test-grid.pkl"), batch_size, label_num=config['grid_size'][0]*config['grid_size'][1]+3, rho=r)
                for a in alpha:
                    for d in distance:
                        abnormal_dataloader = TrajectoryLoader(os.path.join(root_path, f"alpha_{a}_distance_{d}.pkl"), batch_size, label_num=config['grid_size'][0]*config['grid_size'][1]+3, rho=r)
                        
                        pr_auc_list.setdefault(f"a_{a}_d_{d}_r_{r}", list())
                        pr_auc_list[f"a_{a}_d_{d}_r_{r}"].append(trainer.test(normal_dataloader, abnormal_dataloader))


            
            post = f'|{city}'
            for a in alpha:
                for d in distance:
                    for r in ratio:
                        post += f'|{pr_auc_list[f"a_{a}_d_{d}_r_{r}"][-1]}'

            print(post)
            with open("log.txt", 'a') as f:
                f.write(post + '\n')
        
        post = '|Mean'
        for a in alpha:
            for d in distance:
                for r in ratio:
                    post += f'|{round(sum(pr_auc_list[f"a_{a}_d_{d}_r_{r}"])/runs, 4)}'

        print(post)
        with open("log.txt", 'a') as f:
            f.write(post + '\n')