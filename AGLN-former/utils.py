import torch
import torch.nn as nn
import numpy as np
import itertools
from sklearn import manifold
from dataloader import get_file_path
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
import scipy.io
scaler = StandardScaler()

def knn_value(x):
    inner = -2*torch.matmul(x.transpose(3, 2), x)
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(3, 2)
    value = abs(pairwise_distance)
    return value

class EarlyStopping_gan:

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, dep=True, sub='01', clip_num=1):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dep = dep
        self.sub = sub
        self.clip_num = str(clip_num)
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        path_dep = config.save_index_1 + '/checkpoint/subject_dependent/checkpoint_gan_dep_' + self.sub + '_'+'clip' + self.clip_num +'.pt'
        path_indep = config.save_index_1 + '/checkpoint/subject_dependent/checkpoint_gan_indep_' + self.sub + '.pt'
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), path_dep if self.dep else path_indep)
        self.val_loss_min = val_loss


from bisect import bisect_right

class schedule(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get_warm_scheduler(self, milestones, warmup_iters, gamma=0.5):
        return WarmupMultiStepLR(self.optimizer, milestones=milestones, warmup_iters=warmup_iters, gamma=gamma)

    def get_milestone_scheduler(self, milestones, gamma=0.5):
        return torch.optim.lr_scheduler.MultiStepLR(self.optimizer,  milestones=milestones, gamma=gamma, last_epoch=-1)




