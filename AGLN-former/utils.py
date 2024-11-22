import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import config
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

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted",
                " but got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        warmup_factor = 1
        list = {}
        if self.last_epoch < self.warmup_iters:  # 0<10
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor  # 1/3
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                list = {"last_epoch": self.last_epoch, "warmup_iters": self.warmup_iters, "alpha": alpha,
                        'warmup_factor': warmup_factor}

        # print(base_lr  for base_lr in    self.base_lrs)
        # print(base_lr* warmup_factor* self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs)

        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in
                self.base_lrs]
