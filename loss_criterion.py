import torch.nn as nn
import torch


class MaskedBCELogitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self,logits,target):
        mask = (target != -1).float()
        safe_target = torch.where(target == -1,0.5,target)

        loss = self.bce(logits,safe_target) * mask

        num_elements = mask.sum()
        loss = loss.sum() / (num_elements + 1e-8)
        return loss
    