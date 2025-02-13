import torch.nn as nn

def l1_loss(pred, target):
    criterion = nn.L1Loss()
    return criterion(pred, target)
