'''
Created on March 9 2022 14:52:43
@author(s): HuangLab
'''
from collections import OrderedDict
import matplotlib
# matplotlib.use('Agg')  # 绘图不显示

import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, datasets

import time
import datetime

import sys
import configuration as conf

# 把 sys path 添加需要 import 的文件夹
# sys.path.append(conf.ROOT_PATH + 'fedLearn/')


from utils.sampling import mnist_iid, cifar_iid, mnist_noniid, cifar_noniid
from utils.sampling import mnist_iid_modified, cifar_iid_modified, mnist_noniid_modified
from utils.timeSim import Client_Sim, Find_stragglers_id_and_time_thres, Get_local_epoch
from utils.options import args_parser
from utils.write import WriteToTxt
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifarPlus
from models.resnet import ResNet
from models.Fed import FedAvg, FedAvgV1
from models.test import test_img


# def FedLearnSimulate(alg_str='linucb', args_model='cnn', valid_list_path="valid_list_linucb.txt",
#                      args_dataset='mnist', args_usernumber=100, args_iid=False, map_file=None,
#                       threshold_K=7, isKSGD=False, isFullSGD=False):
def FedLearnSimulate(args_model='cnn', args_dataset='mnist', record_name="record", 
                    args_usernumber=10, args_iid=False, map_file=None, 
                    threshold_K=7, strag_h=1, other_class=9, isKSGD=False, isFullSGD=False):
    '''
    这个函数是执行 Federated Learning Simulation 的 main 函数
    '''
    WriteToTxt(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + "\n", record_name)
    # 保存文本的时候使用，此时是 LinUCB 方法
    # result_str = alg_str

    # valid_list = np.loadtxt('noniid_valid/valid_list_fedcs.txt')
    # valid_list = np.loadtxt('valid_list_linucb.txt')                # 这个是 iid 情况的 devices selection 文件
    # valid_list = np.loadtxt(valid_list_path, encoding='utf_8_sig')
    # 10*200 --> 猜测应该是 200 communication rounds，10个设备中每行（即每个round）设备数值不为-1的就可以挑选

    # load args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print("cuda is available : ", torch.cuda.is_available())        # 本次实验使用的 GPU 型号为 RTX 2060 SUPER，内存专用8G、共享8G

    WriteToTxt("K-SGD? {}, Full-SGD? {}\n".format(isKSGD, isFullSGD), record_name)
    WriteToTxt("usernumber is {}, threshold_K is {}, Straggling Period is {}\n".format(args_usernumber, threshold_K, strag_h), record_name)
    WriteToTxt("Model is {}, dataset is {}, one worker has {} other + 1 main classes, is IID? {}\n".format(args_model, args_dataset, other_class, args_iid), record_name)
    WriteToTxt("{} rounds, {} local epoch \n".format(args.epochs, args.local_ep, other_class), record_name)
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

        if args.iid:        # 好像没看见 non-iid 的代码
            print("args.iid is true")
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_iid_modified(dataset_train, args.num_users)
        else:
            print("args.iid is false, non-iid")
            # dict_users = mnist_noniid(dataset_train, args.num_users)
            dict_users = mnist_noniid_modified(dataset_train, args.num_users,
                                               main_label_prop=0.8, other=other_class, map_file=map_file)
            print("args.iid is false, non-iid")
    elif args.dataset == 'cifar':
        print("cifar dataset!")
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            print("cifar iid")
            # dict_users = cifar_iid(dataset_train, args.num_users)
            dict_users = cifar_iid_modified(dataset_train, args.num_users)
        else:
            print("cifar non-iid")
            dict_users = cifar_noniid(dataset_train, args.num_users,
                                      min_train=2000, max_train=2100, main_label_prop=0.8, other=other_class, map_file=map_file)
    else:
        exit('Error: unrecognized dataset')
    # load dataset
    #####################################################################################################################

    #####################################################################################################################
    # time simulate
    time_all_client = []
    for i in range(args.num_users):
        miu_k = args.local_ep * len(dict_users[i]) * 0.5
        client = Client_Sim(datasize=len(dict_users[i]), a_k=0.01, miu_k=miu_k, local_epoch=args.local_ep, rounds=args.epochs)
        time_all_client.append(client.get_execution_time())

    timer = 0
    # time simulate
    #####################################################################################################################

    #####################################################################################################################
    # build model
    print("build model")
    img_size = dataset_train[0][0].shape
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
        # print("resnet:\n", global_net)
    else:
        exit('Error: unrecognized model')
    # build model
    #####################################################################################################################
    #####################################################################################################################

    print("global_net:\n", global_net)
    global_net.train()

    # copy weights
    w_glob = global_net.state_dict()

    # 这几个最后保存为txt
    loss_avg_client  = []
    acc_global_model = []
    valid_number_list= []
    time_rounds = []

    #####################################################################################################################
    #####################################################################################################################
    # training
    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]

    # last_loss_avg = 0
    # last_acc_global = 0
    #####################################################################################################################
    # warm up n_ep epoch
    n_ep = 2
    for round in range(n_ep):
        print("warm-up round {} start:".format(round))

        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_locals_strag = []

        # 全部参与训练
        user_idx_this_round = np.arange(0, args.num_users, 1)
        print("(warm-up)user_idx_this_round:", user_idx_this_round)

        if len(user_idx_this_round) > 0:
            total_data_sum = 0  # 所有设备datasize相加
            for ix in user_idx_this_round:
                total_data_sum += len(dict_users[ix])
            print("total_data_sum: ", total_data_sum)

            # Local Training start
            for idx in user_idx_this_round:     # 遍历可选的设备
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

            # On Server:
            # Global Model Aggregation 
            w_glob = FedAvgV1(w=w_locals, total_data_sum=total_data_sum,
                            user_idx_this_round=user_idx_this_round,
                            dict_users=dict_users)   

            # copy weight to net_glob
            global_net.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            # loss_avg_client.append(loss_avg)
            acc_test, loss_test = test_img(global_net, dataset_test, args)
            # acc_global_model.append(acc_test)
            print('(warm-up): Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, valid {:3d}'
                  .format(round, loss_avg, acc_test, len(user_idx_this_round)))
            WriteToTxt('(warm-up): Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, valid {:3d}'
                  .format(round, loss_avg, acc_test, len(user_idx_this_round)) + "\n", record_name)
    ###################################################################################################
    ###################################################################################################
    # train start (Default is LGC_SGD)
    w_compensation = OrderedDict()
    
    round_h = 0 # 配合 strag_h 形成 Straggling Period，使离群者保持离群 'strag_h' rounds
    print("args.epochs: ", args.epochs)
    for round in range(args.epochs):
        print("round {} start:".format(round))

        loss_locals = []
        if not args.all_clients:
            w_locals = []
            w_locals_strag = []
     
        user_idx = np.arange(0, args.num_users, 1)
        # 1. 离群者随机产生（测试用）
        # user_idx_this_round = np.random.choice(range(args.num_users), threshold_K, replace=False)  # 在num_users里面选K个
        # user_idx_Straggle = np.setdiff1d(user_idx, user_idx_this_round, assume_unique=True)
        # 2. 离群者由模拟时间排序产生，并获得非离群者worker的最慢时间（加上服务器聚合时间即为当前round的时间）
        if round % strag_h == 0:
            round_h = round

        if isFullSGD:
            user_idx_this_round = user_idx
            user_idx_Straggle = []
            _, time_thres = Find_stragglers_id_and_time_thres(times_all=time_all_client, round=round_h, strag_size=0)
        else:
            user_idx_Straggle, time_thres = Find_stragglers_id_and_time_thres(times_all=time_all_client, round=round_h, strag_size=args.num_users-threshold_K)
            user_idx_this_round = np.setdiff1d(user_idx, user_idx_Straggle, assume_unique=True)

        print("user_idx:", user_idx)
        print("user_idx_this_round:", user_idx_this_round)
        print("user_idx_Straggle:", user_idx_Straggle)

        if len(user_idx_this_round) > 0:
            # 计算 datasize
            total_data_sum = 0  # 所有（非离群）设备datasize相加
            total_strag_data_sum = 0  # 所有（离群）设备datasize相加
            for ix in user_idx_this_round:
                total_data_sum += len(dict_users[ix])
            for ix in user_idx_Straggle:
                total_strag_data_sum += len(dict_users[ix])
            print("total_data_sum: ", total_data_sum)
            print("total_strag_data_sum: ", total_strag_data_sum)

            # Local Training start
            for idx in user_idx_this_round:     # 遍历可选的设备
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

            # Straggler Training start  # 与上面的训练代码一样，在train中加了epoch选项
            for idx in user_idx_Straggle:     # 遍历Straggle的设备
                if isKSGD:
                    break;

                print("straggler user {} local training".format(idx))

                strag_local_epoch = Get_local_epoch(times_all=time_all_client, round_time=time_thres, 
                                                    idx=idx, local_ep=args.local_ep, round=round_h)
                local = LocalUpdate(args=args, dataset=dataset_train,
                                    idxs=dict_users[idx])
                weight, loss = local.train(net=copy.deepcopy(global_net).to(args.device), straggle=True, epoch=strag_local_epoch)
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

            if isFullSGD or isKSGD:
                global_net.load_state_dict(w_glob)
            else:
                # LGC:使用 compensation 进行补偿
                if round == 0:
                    w_glob_comped = copy.deepcopy(w_glob)
                else:
                    w_glob_comped = w_Add(w_glob, w_compensation)

                # copy weight to net_glob
                global_net.load_state_dict(w_glob_comped)

                # LGC:依据（上一轮的） w_locals_strag，w_glob 计算出 compensation
                # 算法中属于下一 round 放在此处可以避免更多变量的声明
                w_strag = FedAvgV1(w=w_locals_strag, total_data_sum=total_strag_data_sum,
                                user_idx_this_round=user_idx_Straggle, 
                                dict_users=dict_users) # 离群者训练梯度的求和（datasize加权）平均
                # 1. 不考虑 datasize
                # w_left = w_Mul(float(args.num_users - threshold_K) / args.num_users), w_strag)
                # w_right = w_Mul((float(args.num_users - threshold_K) / args.num_users), w_glob)
                # 2. 考虑 datasize：K -> total_data_sum, N-K -> total_strag_data_sum
                w_left = w_Mul(float(total_strag_data_sum) / (total_data_sum + total_strag_data_sum), w_strag)
                w_right = w_Mul(float(total_strag_data_sum) / (total_data_sum + total_strag_data_sum), w_glob)
                w_compensation = w_Sub(w_left, w_right)
            #####################################
            # train end

            # timer
            timer += time_thres
            time_rounds.append(timer)
            
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_avg_client.append(loss_avg)
            acc_test, loss_test = test_img(global_net, dataset_test, args)
            acc_global_model.append(acc_test)
            # last_loss_avg = loss_avg
            # last_acc_global = acc_test
            if isKSGD:
                print("K-SGD",end=" ")
            elif isFullSGD:
                print("Full-SGD",end=" ")
            else:
                print("LGC-SGD",end=" ")

            print("Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, valid {:3d}, time {:.2f}"
                  .format(round, loss_avg, acc_test, len(user_idx_this_round), timer))
            WriteToTxt("Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, valid {:3d}, time {:.2f}"
                  .format(round, loss_avg, acc_test, len(user_idx_this_round), timer) + "\n", record_name)
            
        # else:
        #     print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f} 0 !'
        #           .format(round, last_loss_avg, last_acc_global))
        #     loss_avg_client.append(last_loss_avg)
        #     acc_global_model.append(last_acc_global)

        # round end
        #############################################################
    # training
    #####################################################################################################################
    #####################################################################################################################

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_avg_client)), loss_avg_client)
    # plt.ylabel('train_loss')
    # plt.savefig('loss_random_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # plt.figure()
    # plt.plot(range(len(acc_global_model)), acc_global_model)
    # plt.ylabel('acc_global')
    # plt.show()
    # plt.savefig('acc_random_{}_{}_E{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # np_valid_number_list = np.array(valid_number_list)


def multiSimulateMain():
    # 参数：1. 三种模式 2. Straggling Period 3. 数据集&网络？ 4.用户数10/20/40... 5. 数据分布iid 
    # args_model='cnn', args_dataset='mnist', record_name="record", args_usernumber=10, args_iid=False, 
    # map_file=None, threshold_K=7, strag_h=1, isKSGD=False, isFullSGD=False)
    FedLearnSimulate(args_dataset='cifar', args_model='resnet', record_name="lgc_NiidcifarRes", args_usernumber=10, args_iid=False, strag_h=3)
    FedLearnSimulate(args_dataset='cifar', args_model='resnet', record_name="K_NiidcifarRes", args_usernumber=10, args_iid=False, strag_h=3, isKSGD=True)
    FedLearnSimulate(args_dataset='cifar', args_model='resnet', record_name="Full_NiidcifarRes", args_usernumber=10, args_iid=False, strag_h=3, isFullSGD=True)

    print("multi-simulation end")

def w_Add(w1, w2):
    w = copy.deepcopy(w1) 
    for k in w.keys():
        w[k] = w[k] + w2[k]
    return w

def w_Sub(w1, w2):
    w = copy.deepcopy(w1)
    for k in w.keys():
        w[k] = w[k] - w2[k]
    return w

def w_Mul(n, w1):
    w = copy.deepcopy(w1)
    for k in w.keys():
        w[k] = torch.mul(w[k], n)
    return w


if __name__ == '__main__':
    multiSimulateMain()
