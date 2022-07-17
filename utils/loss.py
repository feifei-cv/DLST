import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon, eps_vanilla, reduction=True, type=None):

        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.eps_vanilla = eps_vanilla
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction
        self.type = type

    def forward(self, inputs, targets, his_y_s=None):

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        targets = targets.to(device)
        if self.type == 'vanilla':
            targets = (1 - self.eps_vanilla)* targets + self.eps_vanilla / self.num_classes
        else:
            w1 = (1 - self.epsilon).view(-1,1).expand_as(targets)
            w2 = (self.epsilon).view(-1, 1).expand_as(his_y_s)
            targets =  w1 * targets + w2 * his_y_s
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction=='mean':
            return loss.mean()
        else:
            return loss


class DynamicSmooth(nn.Module):

    def __init__(self, num_classes):
        super(DynamicSmooth, self).__init__()
        self.num_classes = num_classes
        self.emd_size = 256
        self.alpha = 0.1
        self.center = torch.rand(size=[self.num_classes, self.emd_size]).to(device)/self.num_classes
        # self.center = torch.zeros([self.num_classes, self.emd_size]).to(device)

    def update_center(self, inputs_row, target_row):

        for i in torch.arange(self.num_classes):
            row_mask = torch.where(target_row == i)
            if len(row_mask[0]) == 0:
                continue
            else:
                self.center[i] = (1-self.alpha)*self.center[i] + self.alpha*torch.mean(inputs_row[row_mask], dim=0)
        return self.center

    def cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return torch.mm(x,  y.T)

    def forward(self, inputs_col, targets_col, inputs_row, target_row, epoch):

        if inputs_row.shape[0] == 0:
            return self.center, torch.zeros_like(targets_col).to(device)
        self.center = self.update_center(inputs_row, target_row)
        dist = self.cosine_dist(self.center, inputs_col)
        all_sim_exp = 1 + torch.exp(dist)
        softmax_loss = all_sim_exp / all_sim_exp.sum(dim=0)
        if epoch == 0:
            smooth_rate = torch.zeros_like(targets_col).to(device)
        else:
            C = [min(torch.exp(softmax_loss[:, i][selected_cls]*epoch)/self.num_classes, 0.1) for i, selected_cls in enumerate(targets_col)]
            smooth_rate = torch.Tensor(C)
        return self.center, smooth_rate



class SemanticLoss(nn.Module):

    def __init__(self, num_classes):

        super(SemanticLoss, self).__init__()
        self.num_classes = num_classes
        self.prior = (torch.ones((self.num_classes, 1)) * (1 / self.num_classes)).to(device) # uniform
        self.eps = 1e-6


    def pairwise_cosine_dist(self, x, y):

        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        dis = 1 - torch.matmul(x, y.T)
        return dis


    def forward(self, center_s, f_t):

        sim_matrix = torch.matmul(center_s, f_t.T)
        new_logits = torch.log(self.prior + self.eps) + sim_matrix.detach()
        s_dist = F.softmax(new_logits, dim=0) ## transport distance
        cost_matrix = self.pairwise_cosine_dist(center_s, f_t) ## cost matrix
        loss = (cost_matrix * s_dist).sum(0).mean()
        return loss



def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return item1 - item2
