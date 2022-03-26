#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torchvision as tv
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, straggle=False, epoch=3):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

        local_ep = self.args.local_ep
        if straggle==True:
            local_ep = epoch

        epoch_loss = []
        print("local epoch is {}".format(local_ep))
        for iter in range(local_ep):
            # print("local epoch {} start".format(iter))
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print("images:\n", type(images), ' | ', images.shape)
                # trans = tv.transforms.ToTensor()  # Jan 18 2022 11:1:30
                # images = trans(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class SGDLocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, straggle=False, pts=3):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

        local_pts = self.args.local_pts
        if straggle==True:
            local_pts = pts

        batch_loss = []
        gradients_local = {}

        batch_num = len(self.ldr_train)
        train_batch_num = math.floor(batch_num * local_pts / self.args.local_pts)
        print("{} / {} data will be trained".format(local_pts, self.args.local_pts))
        print("Data are divided into {} batches, {} will be trained".format(batch_num, train_batch_num))
     
        for batch_idx, (images, labels) in enumerate(self.ldr_train):

            if batch_idx + 1 >= train_batch_num:
                break
            
            net.zero_grad()
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward() 
            # 累加计算出的梯度，时间足够训练的情况下有 len(self.ldr_train) 个
            if batch_idx == 0:
                for name, temp_params in net.named_parameters():
                    # print("grad names:", name)
                    gradients_local[name] = temp_params.grad
            else:
                for name, temp_params in net.named_parameters():
                    gradients_local[name] += temp_params.grad

            ########################################################################
            # optimizer.step()
            # 避免本地进行模型更新
            ########################################################################
            batch_loss.append(loss.item())
        epoch_loss = sum(batch_loss)/len(batch_loss)
        return gradients_local, epoch_loss, batch_num

