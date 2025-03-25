
import torch
#torch.autograd.set_detect_anomaly(True)
from .mixin import Loss


class L2Loss(Loss):

    def forward(self, pred, gt, aggregate=False):
        loss = (gt - pred).pow(2).sum(-1)

        if aggregate:
            return loss.mean()
        return loss

class L1Loss(Loss):

    def forward(self, pred, gt, aggregate=False):
        loss = (gt - pred).abs().sum(-1)
        print('landmarks test',gt.shape)
        if aggregate:
            return loss.mean()
        return loss

