import torch
from torch import nn


class PolarIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,y_pred, y_gt):
        batchsize = y_pred.shape[0]
        dim = y_pred.shape[1]
        loss = 0.0
        for i in range(batchsize):
            sum_max = torch.zeros(1)
            sum_min = torch.zeros(1)
            for j in range(dim):
                sum_max += torch.max(y_pred[i][j], y_gt[i][j])
                sum_min += torch.min(y_pred[i][j], y_gt[i][j])
            loss += torch.log(sum_max/sum_min)
        loss /= batchsize
        return loss

