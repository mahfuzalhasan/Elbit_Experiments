import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

class SegmentationLosses(object):
    def __init__(self, batch_average=True, cuda=True):
        # self.ignore_index = ignore_index
        # self.weight = weight
        # self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='dice'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.DiceLoss
        elif mode == 'dice2':
            return self.DiceLoss2
        else:
            raise NotImplementedError

    

    
    
    def DiceLoss(self, logit, target):

        n, c, t, h, w = logit.size()
        # true_1_hot = torch.eye(c)[target.type(torch.long)]
        # true_1_hot.to(logit.device)
        # true_1_hot = true_1_hot.permute(0, 3, 1, 2)
        # for i in range(n):
        #     if torch.max(target[i]).cpu().item() > 0:
        #         true_1_hot[i,0,:,:] = torch.zeros((h,w))
        smooth = 1

        # probas = F.softmax(logit, dim=1)
        # true_1_hot = true_1_hot.type(probas.type())
        probas = torch.sigmoid(logit)
        true_1_hot = target
        # true_1_hot.to(probas.device)
        # target_inflated = target.unsqueeze(1)
        # logit_3x3 = probas.unfold(2,3,3).unfold(3,3,3).contiguous()
        # logit_3x3 = logit_3x3.view(n, -1, 3, 3)
        logit_5x5 = probas.unfold(3,5,5).unfold(4,5,5).contiguous()
        logit_5x5 = logit_5x5.view(n, -1, 5, 5)
        logit_7x7 = probas.unfold(3,7,7).unfold(4,7,7).contiguous()
        logit_7x7 = logit_7x7.view(n, -1, 7, 7)


        # target_3x3 = true_1_hot.unfold(2,3,3).unfold(3,3,3).contiguous()
        # target_3x3 = target_3x3.view(n, -1, 3, 3)
        target_5x5 = true_1_hot.unfold(3,5,5).unfold(4,5,5).contiguous()
        target_5x5 = target_5x5.view(n, -1, 5, 5)
        target_7x7 = true_1_hot.unfold(3,7,7).unfold(4,7,7).contiguous()
        target_7x7 = target_7x7.view(n, -1, 7, 7)
        # pdb.set_trace()

        def dice_loss(current_logit, current_target, smooth=smooth):
            dims = (0,) + tuple(range(2, current_target.ndimension()))
            intersection = torch.sum(current_logit * current_target, dims)
            cardinality = torch.sum(current_logit + current_target, dims)
            dice_Loss = ((2. * intersection + smooth) / (cardinality + smooth)).mean()
            return 1-dice_Loss


        # loss_3_3 = dice_loss(logit_3x3, target_3x3)
        # pdb.set_trace()
        loss_5_5 = dice_loss(logit_5x5, target_5x5)
        loss_7_7 = dice_loss(logit_7x7, target_7x7)
  
        criterion = nn.BCEWithLogitsLoss()

        if self.cuda:
            criterion = criterion.cuda()

        ce_loss = criterion(logit.view(-1), target.view(-1))

        if self.batch_average:
            ce_loss /= n

        return loss_5_5 + loss_7_7 + 2*ce_loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 1, 8, 224, 400).cuda()
    b = torch.rand(1, 1, 8, 224, 400).cuda()
    print(loss.DiceLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




