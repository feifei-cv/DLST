import torch
from typing import Optional
from torch.optim.optimizer import Optimizer


log_info = dict()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class memoryBank:

    def __init__(self,num_classes, bottleneck_dim):

        super(memoryBank, self).__init__()
        self.K = 8192
        self.num_classes = num_classes
        self.feats = torch.zeros(self.K, bottleneck_dim).to(device)
        self.targets = torch.zeros(self.K, dtype=torch.long).to(device)
        self.predict = torch.zeros(self.K, self.num_classes).to(device)
        self.targets[:]=-1
        self.ptr = 0
        self.sigma= 0.1


    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets, y_s):
        q_size = len(targets)
        if self.ptr + q_size > self.K:

            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.predict[-q_size:] = y_s
            self.ptr = 0
            # self.ptr = self.ptr % self.K
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.predict[self.ptr: self.ptr + q_size] = y_s
            self.ptr += q_size


    def cum(self, label_s, y_s): ## current batch

        self.y_new = torch.zeros_like(y_s)
        for index, i in enumerate(label_s):
            row_mask = torch.where(self.targets == i) # retrive
            if len(row_mask[0]) == 0:
                self.y_new[index] = y_s[index].clone()
            else:
                self.y_new[index] =  (1-self.sigma) * torch.mean(self.predict[row_mask], dim=0) + \
                                     self.sigma * y_s[index].clone()
        return self.y_new



class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:
    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},
    where `i` is the iteration steps.
    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.0002, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()

        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                if 'lr_mult' not in param_group:
                    param_group['lr_mult'] = 1.
                param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1





