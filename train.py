import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import CDPNet, combined_loss
from dataset import DPDataset
from filehelp import date_string
from readconfig import readconfig

def save_train_history(train, val, figname, val_freq):
    train_line, = plt.plot(train, label='Training Loss')
    val_line, = plt.plot(val, label='Validation Loss')
    plt.xlabel(' x {} Batches'.format(val_freq))
    plt.ylabel('Loss')
    plt.legend()
    #plt.ylim((-2,6))
    plt.savefig(figname)

def main():
    '''get parameters'''
    func_dict, dims_dict, scales_dict, params_dict = readconfig(INIPATH)

    '''setup data'''
    train_set = DPDataset(
        DPATH,
        scales_dict,
        seq_len=int(params_dict['seq_len']),
        separation=int(params_dict['separation'])
    )
    train_loader = DataLoader(train_set, batch_size=int(params_dict['batch']), shuffle=True, drop_last=True)

    val_set = DPDataset(DPATH, scales_dict=scales_dict, seq_len=int(params_dict['seq_len']), val=True)
    val_loader = DataLoader(val_set, batch_size=int(params_dict['batch']), shuffle=True, drop_last=True)
    iter_val = val_loader.__iter__()

    '''setup model'''
    model = CDPNet(funcs_dict=func_dict, dims_dict=dims_dict)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        #print(isinstance(model, nn.DataParallel))
    model.to(DEVICE)
    opt = torch.optim.RMSprop(model.parameters(), lr=params_dict['lr_init'])
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, cooldown=5, min_lr=1e-6)

    '''setup checkpoint directory'''
    weights_path = './weights'
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    save_path = os.path.join(weights_path,date_string())
    os.mkdir(save_path)

    config_name = os.path.split(INIPATH)[-1]
    copyfile(INIPATH, os.path.join(save_path, config_name))
    print('saving weights to', save_path)

    '''training'''
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    val_freq = 10
    save_freq = 100
    total_batches = len(train_loader)

    start = time.time()
    try:
        for e in range(params_dict['epoch']):
            for b, batch in enumerate(train_loader):
                X = batch[0]
                y = batch[1]
                X = [x.float().to(DEVICE) for x in X]
                y = [wy.float().to(DEVICE) for wy in y]

                pred, hidden_states = model(X)

                total_loss, s_loss, p_loss = combined_loss(pred, y, X[2][:,-1,:])
                total_loss.backward()
                opt.step()
                opt.zero_grad()

                if (b%save_freq==0):
                    if b!=0:
                        if isinstance(model, nn.DataParallel):
                            sd = model.module.state_dict()
                        else:
                            sd = model.state_dict()

                        checkpoint_dir = os.path.join(save_path, 'e{}b{}.pt'.format(e,b))
                        torch.save(sd, checkpoint_dir)


                if (b%val_freq==0) and b!=0:
                    model.eval()
                    try:
                        X_val, y_val = next(iter_val)
                    except StopIteration:
                        #if we reach the end of validation data
                        iter_val = val_loader.__iter__()
                        X_val, y_val = next(iter_val)

                    X_val = [x.float().to(DEVICE) for x in X_val]
                    y_val = [wy.float().to(DEVICE) for wy in y_val]

                    with torch.no_grad():
                        pred_val, hidden_val = model(X_val)
                        loss_val, s_loss_val, p_loss_val = combined_loss(pred_val, y_val, X_val[2][:,-1,:])

                    val_loss_history.append(loss_val.item())
                    train_loss_history.append(total_loss.item())

                    print('e {}, b {}/{}, loss: {:.3f}, s_loss: {:.3f}, p_loss: {:.3f}, val loss: {:.3f}'.format(e, b, total_batches, total_loss, s_loss, p_loss, loss_val))

                    model.train()

                #if b>=20:
                    #break
    finally:
        if isinstance(model, nn.DataParallel):
            sd = model.module.state_dict()
        else:
            sd = model.state_dict()
        checkpoint_dir = os.path.join(save_path, 'final.pt')
        torch.save(model.state_dict(), checkpoint_dir)

        fig_path = os.path.join(save_path, 'loss.png')
        save_train_history(train_loss_history, val_loss_history, fig_path, val_freq)
        os.system('nvidia-smi')



if __name__=='__main__':
    #get data path and config path
    DPATH = sys.argv[1]
    INIPATH = sys.argv[2]
    print('Training with parameters from', INIPATH)
    main()
