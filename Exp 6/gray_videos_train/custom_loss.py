import torch
import torch.nn as nn

import numpy as np

class WeightedBCELoss(nn.Module):
    # Args:
    #   weights: ndarry vector with dimension #class
    #   non_eps: replace 0 weights with this value. If it is zero, it doesn't change anything
    def __init__(self, use_cuda, weights=None):
        super(WeightedBCELoss, self).__init__()
        #self.weights = crieterion
        if weights is not None:
            self.weights = weights
            if use_cuda:
                self.weights = self.weights.float().cuda()
        else:
            weights = np.ones(2)
            weights = torch.from_numpy(weights)
            self.weights = weights

            if use_cuda:
                self.weights = self.weights.float().cuda()
             
    def forward(self, output, target):
        
        eps = 1e-12    
        losses = []
        
        loss_per_sample = []        

        batch = output.size(0)
        class_no = output.size(1)
        for b in range(batch):
            loss_per_class = []            
            #size = output[b,:,:].size()
            for i in range(class_no):
                loss = self.weights[i] * ((target[b,i] * torch.log(output[b,i]+eps)) + \
                             ((1 - target[b,i]) * torch.log(1 - output[b,i]+eps)))
                
                loss_per_class.append(loss)

            sum_over_per_loss = torch.stack(loss_per_class, dim=0).sum(dim=0)
            loss_per_sample.append(sum_over_per_loss)
            
            
            
        losses = torch.stack([loss for loss in loss_per_sample],dim=0)
        
        return torch.neg(losses)


class WeightedTwoPartBCELoss(nn.Module): 
    def __init__(self, use_cuda, weights=None, neg_weights=None):
        super(WeightedTwoPartBCELoss, self).__init__()
        if weights is not None:
            self.weights = weights
            self.neg_weights = neg_weights
            
        else:
            weights = np.ones(2)
            weights = torch.from_numpy(weights)

            neg_weights = np.ones(2)
            neg_weights = torch.from_numpy(neg_weights)

            self.weights = weights
                
        if use_cuda:
            self.weights = self.weights.float().cuda()
            self.neg_weights = self.neg_weights.float().cuda()

             
    def forward(self, output, target):
        
        eps = 1e-12    
        losses = []
        
        loss_per_sample = []        

        batch = output.size(0)
        class_no = output.size(1)
        for b in range(batch):
            loss_per_class = []            
            #size = output[b,:,:].size()
            for i in range(class_no):
                loss = self.weights[i] * (target[b,i] * torch.log(output[b,i]+eps)) + self.neg_weights[i] * ((1 - target[b,i]) * torch.log(1 - output[b,i]+eps))
                loss_per_class.append(loss)
            
            
            sum_over_per_loss = torch.stack(loss_per_class, dim=0).sum(dim=0)
            
            loss_per_sample.append(sum_over_per_loss)
            
        losses = torch.stack([loss for loss in loss_per_sample],dim=0)
        return torch.neg(losses)


class DiceLoss(nn.Module):
    def __init__(self, use_cuda):
        super(DiceLoss, self).__init__()
        self.use_cuda = use_cuda


    def forward(self, output, targets, eps=1e-12):
        smooth = 1
        num = targets.size(0)
        probs = output
        
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
