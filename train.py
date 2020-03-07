import os
import sys
import time
from configparser import ConfigParser
from shutil import copyfile

import matplotlib.pyplot as plt
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
#function alias for imported models
resnet18 = torchvision.models.resnet18

from model import return_2_fc, return_lstm, return_branch, CDPNet
from dataset import DPDataset
from filehelp import date_string

def main():
    '''get parameters'''
    parser = ConfigParser()
    parser.read(INIPATH)
    #get list of keys for each group of variables
    modules_list = parser.options('modules')
    dims_list = parser.options('dims')
    scales_list = parser.options('scales')
    params_list = parser.options('params')

    func_dict = {}
    g = globals()
    for m_key in modules_list:
        func_key = parser.get('modules', m_key)
        func_dict[m_key] = g[func_key]

    dims_dict = {}
    for d_key in dims_list:
        dim = parser.get('dims', d_key)
        dims_dict[d_key] = int(dim)

    scales_dict = {}
    for s_key in scales_list:
        scale = parser.get('scales', s_key)
        scales_dict[s_key] = float(scale)

    params_dict = {}
    for p_key in params_list:
        param = parser.get('params', p_key)
        params_dict[p_key] = param

    '''setup data'''
    train_set = DPDataset(
        DPATH,
        scales_dict,
        seq_len=int(params_dict['seq_len']),
        separation=int(params_dict['separation'])
    )
    train_loader = DataLoader(train_set, batch_size=int(params_dict['batch']), shuffle=True)

    val_set = DPDataset(DPATH, scales_dict=scales_dict, seq_len=int(params_dict['seq_len']), val=True)
    val_loader = DataLoader(val_set, batch_size=int(params_dict['batch']), shuffle=True)
    iter_val = val_loader.__iter__()

    '''setup model'''
    model = CDPNet(func_dict=func_dict, dims_dict=dims_dict)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        #print(isinstance(model, nn.DataParallel))
    model.to(DEVICE)

    '''setup checkpoint directory'''
    weights_path = './weights'
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    save_path = os.path.join(weights_path,date_string())
    os.mkdir(save_path)
    copyfile(INIPATH, os.path.join(save_path, INIPATH))
    print('saving weights to', save_path)


if __name__=='__main__':
    #get data path and config path
    DPATH = sys.argv[1]
    INIPATH = sys.argv[2]
    print('Training with parameters from', INIPATH)
    main()
