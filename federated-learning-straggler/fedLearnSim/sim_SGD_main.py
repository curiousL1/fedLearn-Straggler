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
from utils.tools import WriteToTxt, W_Add, W_Mul, W_Sub
from models.Update import SGDLocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifarPlus
from models.resnet import ResNet
from models.Fed import FedSGD
from models.test import test_img


def FedLearnSimulate(args_model='cnn', args_dataset='mnist', record_name="record", 
                    args_usernumber=10, args_iid=False, map_file=None, 
                    threshold_K=7, strag_h=1, other_class=9, isKSGD=False, isFullSGD=False, hasTimer=True):
    '''
    这个函数是执行 FedSGD Learning Simulation 的 main 函数
    '''
    WriteToTxt(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + "\n", record_name)

    # load args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.model = args_model
    args.dataset    = args_dataset
    args.num_users  = args_usernumber
    args.iid        = args_iid                 # non-iid

    print("cuda is available : ", torch.cuda.is_available())        # 本次实验使用的 GPU 型号为 RTX 2060 SUPER，内存专用8G、共享8G

    WriteToTxt("Using FedSGD training \n", record_name)
    WriteToTxt("K-SGD? {}, Full-SGD? {}\n".format(isKSGD, isFullSGD), record_name)
    WriteToTxt("usernumber is {}, threshold_K is {}, Straggling Period is {}\n".format(args_usernumber, threshold_K, strag_h), record_name)
    WriteToTxt("Model is {}, dataset is {}, one worker has {} other + 1 main classes, is IID? {}\n".format(args_model, args_dataset, other_class, args_iid), record_name)
    WriteToTxt("{} rounds, {} local partitions \n".format(args.epochs, args.local_pts), record_name)
    print("load dataset")
    
    #####################################################################################################################
    #####################################################################################################################
    # load dataset 
    if args.dataset == 'mnist':
        print("mnist dataset!")
        trans_mnist     = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train   = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test    = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        if args.iid:        # 好像没看见 non-iid 的代码
            print("args.iid is true")
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_iid_modified(dataset_train, args.num_users)
        else:
            print("args.iid is false, non-iid")
            # dict_users = mnist_noniid(dataset_train, args.num_users)
            dict_users = mnist_noniid_modified(dataset_train, args.num_users, min_train=2000, max_train=2000,
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
                                      min_train=2000, max_train=2000, main_label_prop=0.8, other=other_class, map_file=map_file)
    else:
        exit('Error: unrecognized dataset')
    # load dataset
    #####################################################################################################################

    #####################################################################################################################
    # time simulate
    time_all_client = []
    for i in range(args.num_users):
        miu_k = 1 * len(dict_users[i]) * 0.5
        client = Client_Sim(datasize=len(dict_users[i]), a_k=0.01, miu_k=miu_k, local_epoch=1, rounds=args.epochs)
        time_all_client.append(client.get_execution_time())

    timer = 0
    # time simulate
    #####################################################################################################################

    #####################################################################################################################
    # build model
    print("build model")
    img_size = dataset_train[0][0].shape
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
    time_rounds = []

    # train start
    optimizer = torch.optim.SGD(global_net.parameters(), lr=args.lr, momentum=args.momentum)
    ###################################################################################################
    # warm-up
    n_ep = 2
    for round in range(n_ep):
        print("warm-up round {} start:".format(round))

        loss_locals = []
        if not args.all_clients:
            grad_locals = []

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
                local = SGDLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                grad, loss, batch_num = local.train(net=copy.deepcopy(global_net).to(args.device)) 
                grad = W_Mul(1.0 / batch_num, grad) # local update 的时候梯度累加后没有进行平均
                if args.all_clients:
                    grad_locals[idx] = copy.deepcopy(grad)
                else:
                    grad_locals.append(copy.deepcopy(grad))      # append    # 根据user_idx_this_round的顺序 append 上去
                loss_locals.append(copy.deepcopy(loss))
            # Local Training end

            # On Server:
            # Global Gradient Aggregation 
            g_glob = FedSGD(g=grad_locals)

            optimizer.zero_grad()   # 清空 global_net 的梯度
            for name, temp_params in global_net.named_parameters():
                temp_params.grad = g_glob[name] # 梯度更新
            optimizer.step() # 模型更新
            
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
    round_h = 0 # 配合 strag_h 形成 Straggling Period，使离群者保持离群 'strag_h' rounds
    print("args.epochs: ", args.epochs)
    for round in range(args.epochs):
        print("round {} start:".format(round))

        loss_locals = []
        if not args.all_clients:
            grad_locals = []
            grad_locals_strag = []
     
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

                local = SGDLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                grad, loss, batch_num = local.train(net=copy.deepcopy(global_net).to(args.device))
                grad = W_Mul(1.0 / batch_num, grad) # local update 的时候梯度每个partition求和后没有进行平均
                if args.all_clients:
                    grad_locals[idx] = copy.deepcopy(grad)
                else:
                    grad_locals.append(copy.deepcopy(grad))      # append    # 根据user_idx_this_round的顺序 append 上去
                loss_locals.append(copy.deepcopy(loss))
            # Local Training end

            # Straggler Training start  # 与上面的训练代码一样，在train中加了epoch选项
            for idx in user_idx_Straggle:     # 遍历Straggle的设备
                if isKSGD:
                    break;

                print("straggler user {} local training".format(idx))

                strag_local_pts = Get_local_epoch(times_all=time_all_client, round_time=time_thres, 
                                                    idx=idx, local_ep=args.local_pts, round=round_h)
                local = SGDLocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                grad, loss, batch_num = local.train(net=copy.deepcopy(global_net).to(args.device), straggle=hasTimer, pts=strag_local_pts)
                grad = W_Mul(1.0 / batch_num, grad) # local update 的时候梯度每个partition求和后没有进行平均
                if args.all_clients:
                    grad_locals_strag[idx] = copy.deepcopy(grad)
                else:
                    grad_locals_strag.append(copy.deepcopy(grad))     
                loss_locals.append(copy.deepcopy(loss)) ############ loss_local是否需要为Straggler新建?
            # Straggler Training end

            # On Server:
            # Global Model Aggregation 
            g_glob = FedSGD(g=grad_locals)
            
            optimizer.zero_grad()
            if isFullSGD or isKSGD:
                # g_glob = W_Mul(args.lr, g_glob)
                # w_glob = W_Sub(w_glob, g_glob)
                # global_net.load_state_dict(w_glob)
                for name, temp_params in global_net.named_parameters():
                    temp_params.grad = g_glob[name]
                optimizer.step()
            else:
                # LGC:使用 compensation 进行补偿
                if round == 0:
                    g_glob_comped = copy.deepcopy(g_glob)
                else:
                    g_glob_comped = W_Add(g_glob, g_compensation)

                # copy weight to net_glob
                # g_glob = W_Mul(args.lr, g_glob)
                # w_glob = W_Sub(w_glob, g_glob_comped)
                # global_net.load_state_dict(w_glob)
                for name, temp_params in global_net.named_parameters():
                    temp_params.grad = g_glob_comped[name]
                optimizer.step()

                # LGC:依据（上一轮的） grad_locals_strag，w_glob 计算出 compensation
                # 算法中属于下一 round，放在此处可以避免更多变量的声明
                g_strag = FedSGD(g=grad_locals_strag) # 离群者训练梯度的求和
                # 1. 不考虑 datasize
                g_left = W_Mul((float(args.num_users - threshold_K) / args.num_users), g_strag)
                g_right = W_Mul((float(args.num_users - threshold_K) / args.num_users), g_glob)
                # 2. 考虑 datasize：K -> total_data_sum, N-K -> total_strag_data_sum
                # w_left = W_Mul(float(total_strag_data_sum) / (total_data_sum + total_strag_data_sum), w_strag)
                # w_right = W_Mul(float(total_strag_data_sum) / (total_data_sum + total_strag_data_sum), w_glob)
                g_compensation = W_Sub(g_left, g_right)
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
    # save data as file .npy
    np_time_rounds = np.array(time_rounds)
    np_loss_avg_client = np.array(loss_avg_client)
    np_acc_global_model = np.array(acc_global_model)
    np.save(record_name+"_time.npy", np_time_rounds)
    np.save(record_name+"_loss.npy", np_loss_avg_client)
    np.save(record_name+"_acc.npy", np_acc_global_model)

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


def multiSimulateMain(c=10, h=1, dataset='cifar', model='resnet', timer=True, user=10, iid=False):
    # e.g. args_model='cnn', args_dataset='mnist', record_name="record", args_usernumber=10, args_iid=False
    # e.g. map_file=None, threshold_K=7, strag_h=1, isKSGD=False, isFullSGD=False, hasTimer=True
    
    # 记录文件名  (+.txt/.npy)
    other_c = c - 1
    argsStr = "SGD" + dataset + model + 'C' + str(c) + 'H' + str(h) + 'N' + str(user)
    if timer:
        argsStr = argsStr + "Timer"
    if iid:
        argsStr = argsStr + "IID"
    lgc_rc = "lgc_" + argsStr
    k_rc = "k_" + argsStr
    full_rc = "full_" + argsStr
    FedLearnSimulate(args_dataset=dataset, args_model=model, record_name=lgc_rc, args_usernumber=user, args_iid=iid, strag_h=h, other_class=other_c, hasTimer=timer)
    FedLearnSimulate(args_dataset=dataset, args_model=model, record_name=k_rc, args_usernumber=user, args_iid=iid, strag_h=h, other_class=other_c, isKSGD=True, hasTimer=timer)
    FedLearnSimulate(args_dataset=dataset, args_model=model, record_name=full_rc, args_usernumber=user, args_iid=iid, strag_h=h, other_class=other_c, isFullSGD=True, hasTimer=timer)
    print(argsStr + " end")


if __name__ == '__main__':
    #################################### Exp.1
    # multiSimulateMain(c=1, h=10, dataset='cifar', model='resnet', timer=False, user=10, iid=False) # keep wrong
    multiSimulateMain(c=3, h=10, dataset='cifar', model='resnet', timer=False, user=10, iid=False)
    multiSimulateMain(c=10, h=10, dataset='cifar', model='resnet', timer=False, user=10, iid=False)
    multiSimulateMain(c=5, h=1, dataset='cifar', model='resnet', timer=False, user=10, iid=False)
    multiSimulateMain(c=5, h=10, dataset='cifar', model='resnet', timer=False, user=10, iid=False)
    multiSimulateMain(c=5, h=30, dataset='cifar', model='resnet', timer=False, user=10, iid=False)

    # multiSimulateMain(c=1, h=1, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=3, h=1, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=5, h=1, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=10, h=1, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=1, h=10, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=3, h=10, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=5, h=10, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=10, h=10, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=1, h=30, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=3, h=30, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=5, h=30, dataset='cifar', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=10, h=30, dataset='cifar', model='cnn', timer=False, user=10, iid=False)

    # multiSimulateMain(c=1, h=1, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=3, h=1, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=5, h=1, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=10, h=1, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=1, h=10, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=3, h=10, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    multiSimulateMain(c=5, h=10, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=10, h=10, dataset='mnist', model='cnn', timer=False, user=10, iid=False)
    # multiSimulateMain(c=10, h=30, dataset='mnist', model='cnn', timer=False, user=10, iid=False)

    ##########################done
    
