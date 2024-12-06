import torch

def MAELoss(output, target):
    loss = torch.mean(torch.abs((output - target)))
    return loss