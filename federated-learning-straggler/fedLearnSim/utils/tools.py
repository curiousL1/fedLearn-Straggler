import copy
import torch


def WriteToTxt(string, filename, filepath=""):
    file = open(filepath+filename+".txt", 'a')
    file.write(string)
    file.close()

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