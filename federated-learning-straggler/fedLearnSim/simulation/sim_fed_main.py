'''
Created on March 9 2022 14:52:43
@author(s): HuangLab
'''

from typing import OrderedDict
import matplotlib
# matplotlib.use('Agg')  # 绘图不显示

import configuration as conf
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, datasets

import time

import sys
import configuration as conf

# 把 sys path 添加需要 import 的文件夹
# sys.path.append(conf.ROOT_PATH + 'fedLearn/')
sys.path.append(conf.ROOT_PATH)
# print(sys.path)

from fedLearnSim.utils.sampling import mnist_iid, cifar_iid, mnist_noniid, cifar_noniid
from fedLearnSim.utils.sampling import mnist_iid_modified, cifar_iid_modified, mnist_noniid_modified
from fedLearnSim.utils.options import args_parser
from fedLearnSim.models.Update import LocalUpdate
from fedLearnSim.models.Nets import MLP, CNNMnist, CNNCifar, CNNCifarPlus
from fedLearnSim.models.resnet import ResNet
from fedLearnSim.models.Fed import FedAvg, FedAvgV1
from fedLearnSim.models.test import test_img


def FedLearnSimulate(alg_str='linucb', args_model='cnn', valid_list_path="valid_list_linucb.txt",
                     args_dataset='mnist', args_usernumber=100, args_iid=False, map_file=None, threshold_K=7):
    '''
    这个函数是执行 Federated Learning Simulation 的 main 函数
    '''
    # 保存文本的时候使用，此时是 LinUCB 方法
    result_str = alg_str

    # valid_list = np.loadtxt('noniid_valid/valid_list_fedcs.txt')
    # valid_list = np.loadtxt('valid_list_linucb.txt')                # 这个是 iid 情况的 devices selection 文件
    # valid_list = np.loadtxt(valid_list_path, encoding='utf_8_sig')
    # 10*200 --> 猜测应该是 200 communication rounds，10个设备中每行（即每个round）设备数值不为-1的就可以挑选

    # load args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load args

    # 调参
    args.local_bs = 20
    args.local_ep = 4

    print("cuda is available : ", torch.cuda.is_available())        # 本次实验使用的 GPU 型号为 RTX 2060 SUPER，内存专用8G、共享8G

    print("load dataset")
    #####################################################################################################################
    #####################################################################################################################
    # load dataset
    args.dataset    = args_dataset
    args.num_users  = args_usernumber
    args.iid        = args_iid                 # non-iid
    if args.dataset == 'mnist':
        print("mnist dataset!")
        trans_mnist     = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train   = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test    = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users (100)
        # args.iid = True
        if args.iid:        # 好像没看见 non-iid 的代码
            print("args.iid is true")
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_iid_modified(dataset_train, args.num_users)
        else:
            print("args.iid is false, non-iid")
            # dict_users = mnist_noniid(dataset_train, args.num_users)
            dict_users = mnist_noniid_modified(dataset_train, args.num_users,
                                               main_label_prop=0.8, other=9, map_file=map_file)
            print("args.iid is false, non-iid")
    elif args.dataset == 'cifar':
        print("cifar dataset!")
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            print("cifar iid")
            # dict_users = cifar_iid(dataset_train, args.num_users)
            dict_users = cifar_iid_modified(dataset_train, args.num_users)
            # print("dict_users: ", type(dict_users))
            # print(dict_users)
        else:
            print("cifar non-iid")
            dict_users = cifar_noniid(dataset_train, args.num_users,
                                      min_train=200, max_train=1000, main_label_prop=0.8, other=9, map_file=map_file)
    else:
        exit('Error: unrecognized dataset')
    # load dataset
    #####################################################################################################################
    #####################################################################################################################

    img_size = dataset_train[0][0].shape

    print("build model")
    #####################################################################################################################
    #####################################################################################################################
    # build model
    args.model = args_model
    if args.model == 'cnn' and args.dataset == 'cifar':
        print("cnn & cifar")
        # global_net = CNNCifar(args=args).to(args.device)
        global_net = CNNCifarPlus(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        print("cnn & mnist")
        args.num_channels = 1
        global_net = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        print("mlp & cifar")
        len_in = 1
        for x in img_size:
            len_in *= x
        global_net = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        global_net = ResNet(18, num_classes=10).to(args.device)
        print("resnet:\n", global_net)
    else:
        exit('Error: unrecognized model')
    # build model
    #####################################################################################################################
    #####################################################################################################################

    print("global_net:\n", global_net)

    global_net.train()

    # start time
    # time_start = time.time()

    # copy weights
    w_glob = global_net.state_dict()
    w_compensation = OrderedDict()

    # 这几个最后保存为txt
    loss_avg_client  = []
    acc_global_model = []
    valid_number_list     = []

    #####################################################################################################################
    #####################################################################################################################
    # training
    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]

    # last_loss_avg = 0
    # last_acc_global = 0

    args.epochs = 3           # 默认是10，现在设置成200，测试一下 cifar-10 准确率能到多少——Jan 17 2022 23:19:50
    print("args.epochs: ", args.epochs)                  # default --> 10
    for round in range(args.epochs):
        print("round {} start:".format(round))

        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_locals_strag = []
        # round_idx = valid_list[round]       # valid_list 两百行 --> 200 round      即 round_idx是单独一行（one round）的数据（10个数值）
        # user_idx_this_round = round_idx[np.where(round_idx != -1)]      # 一行数据中，等于-1的不选，其它的选上
        # print("user_idx_this_round:\n", user_idx_this_round)

        # 随机产生离群者
        user_idx = np.arange(0, args.num_users, 1)
        user_idx_this_round = np.random.choice(range(args.num_users), threshold_K, replace=False)  # 在num_users里面选K个
        user_idx_Straggle = np.setdiff1d(user_idx, user_idx_this_round, assume_unique=True)
        
        print("user_idx:", user_idx)
        print("user_idx_this_round:", user_idx_this_round)
        print("user_idx_Straggle:", user_idx_Straggle)

        # print("dict_user:\n", type(dict_users), '\n', dict_users)
        # total_data_sum = 0      # 所有设备datasize相加
        # for i in range(len(dict_users)):        # 每行dict_user[idx]的长度，就代表了数据量
        #     total_data_sum += len(dict_users[i])
        # valid_number_list.append(user_idx_this_round)   # 每轮valid participants 数量
        # print("user_idx_this_round:\n", user_idx_this_round)
        if len(user_idx_this_round) > 0:
            total_data_sum = 0  # 所有（非离群）设备datasize相加
            total_strag_data_sum = 0  # 所有（非离群）设备datasize相加
            for ix in user_idx_this_round:
                total_data_sum += len(dict_users[ix])
            for ix in user_idx_Straggle:
                total_strag_data_sum += len(dict_users[ix])
            print("total_data_sum: ", total_data_sum)
            print("total_strag_data_sum: ", total_strag_data_sum)

            # Local Training start
            for idx in user_idx_this_round:     # 遍历可选的设备
                # local updates?
                # print("dataset_train: ", dataset_train.shape)

                # print("dict_user[%d]: ", idx, dict_users[idx])      # 每行[idx]的elements个数不定，一维的list
                print("user {} local training".format(idx))

                local = LocalUpdate(args=args, dataset=dataset_train,
                                    idxs=dict_users[idx])
                weight, loss = local.train(net=copy.deepcopy(global_net).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(weight)
                else:
                    w_locals.append(copy.deepcopy(weight))      # append    # 根据user_idx_this_round的顺序 append 上去
                loss_locals.append(copy.deepcopy(loss))
            # Local Training end

            # Straggler Training start  # 与上面（原本）的训练代码一样，在train中加了epoch选项
            for idx in user_idx_Straggle:     # 遍历Straggle的设备

                # print("dict_user[%d]: ", idx, dict_users[idx])
                print("straggler user {} local training".format(idx))

                local = LocalUpdate(args=args, dataset=dataset_train,
                                    idxs=dict_users[idx])
                weight, loss = local.train(net=copy.deepcopy(global_net).to(args.device), straggle=True, epoch=3) # global_net未更新，且只训练 2 epoch
                if args.all_clients:
                    w_locals_strag[idx] = copy.deepcopy(weight)
                else:
                    w_locals_strag.append(copy.deepcopy(weight))     
                loss_locals.append(copy.deepcopy(loss)) ############ loss_local是否需要为Straggler新建?
            # Straggler Training end

            # On Server:
            # Global Model Aggregation 
            w_glob = FedAvgV1(w=w_locals, total_data_sum=total_data_sum,
                            user_idx_this_round=user_idx_this_round,
                            dict_users=dict_users)   

            #########################################
            # todo:使用 compensation 进行补偿
            if round == 0:
                w_glob_comped = copy.deepcopy(w_glob)
            else:
                w_glob_comped = W_Add(w_glob, w_compensation)
            #########################################
            # w_glob_comped = w_glob # test K-SGD

            # copy weight to net_glob
            global_net.load_state_dict(w_glob_comped)

            #####################################
            # todo:依据（上一轮的） w_locals_strag，w_glob 计算出 compensation
            w_strag = FedAvgV1(w=w_locals_strag, total_data_sum=total_strag_data_sum,
                            user_idx_this_round=user_idx_Straggle, 
                            dict_users=dict_users)
            w_left = W_Mul(1.0 / args.num_users, w_strag)
            w_right = W_Mul(float((args.num_users - threshold_K) / args.num_users), w_glob)
            w_compensation = W_Sub(w_left, w_right)
            #####################################
            
                     
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_avg_client.append(loss_avg)
            acc_test, loss_test = test_img(global_net, dataset_test, args)
            acc_global_model.append(acc_test)
            # last_loss_avg = loss_avg
            # last_acc_global = acc_test
            print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, valid {:3d}'
                  .format(round, loss_avg, acc_test, len(user_idx_this_round)))
        # else:
        #     print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f} 0 !'
        #           .format(round, last_loss_avg, last_acc_global))
        #     loss_avg_client.append(last_loss_avg)
        #     acc_global_model.append(last_acc_global)

    # training
    #####################################################################################################################
    #####################################################################################################################


    # time_end = time.time()
    # print('totally cost time: {:3f}s'.format(time_end - time_start))

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_avg_client)), loss_avg_client)
    # plt.ylabel('train_loss')
    # plt.savefig('loss_random_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    plt.figure()
    plt.plot(range(len(acc_global_model)), acc_global_model)
    plt.ylabel('acc_global')
    plt.show()
    # plt.savefig('acc_random_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # np_valid_number_list = np.array(valid_number_list)


def multiSimulateMain():

    # 设置一共10个，然后随机从10个里面选7个
    FedLearnSimulate(args_dataset='cifar', args_model='resnet', args_usernumber=10, args_iid=False)

    print("multi-simulation end")

def W_Add(w1, w2):
    w = copy.deepcopy(w1) 
    for k in w.keys():
        w[k] = w[k] + w2[k]
    return w

def W_Sub(w1, w2):
    w = copy.deepcopy(w1)
    for k in w.keys():
        w[k] = w[k] - w2[k]
    return w

def W_Mul(n, w1):
    w = copy.deepcopy(w1)
    for k in w.keys():
        w[k] = torch.mul(w[k], n)
    return w


if __name__ == '__main__':
    multiSimulateMain()
