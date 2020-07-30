import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
#import videotransforms
import torch
import torchvision.utils as vutil


#from i3d import InceptionI3d
from i3d_feature_pooling import InceptionI3d
from model_classification import I3D
from loss import *
#from dataloader_feature_pooling import *
#from dataloader_feature_pooling_augmentation import *
from dataloader_loc_feature_pooling_augmentation import *
import sys


import os
import numpy as np
from datetime import datetime
import argparse
import time
#import parameters as params 
from tensorboardX import SummaryWriter
import pickle
import configuration as cfg
import parameters as params

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

#logger = HistorySaver('_results/logs/i3d.npz')
batch_size_local = 5

#f_w = open("/home/c3-0/mahfuz/MEVA_Scripts/class_using_targetted_feature_pooling/file_output/targets_fixed_lr_bce_03-17-20_1416_finetune_rough.txt", "w+")

def visualize(inputs, tubes, run_id, epoch, i, valid = False):
    input_sv_shape = inputs.permute(0,2,1,3,4).shape
    vis_input = inputs.permute(0,2,1,3,4)#.contiguous().view(-1, input_sv_shape[2], input_sv_shape[3], input_sv_shape[4])
    save_path = os.path.join( params.output_dir, run_id )
    if not valid:
        save_path = os.path.join(save_path,'train')
    else:
        save_path = os.path.join(save_path,'valid')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vis_input = vis_input.data.cpu().numpy()
    vis_input = np.transpose(vis_input,(0,1,3,4,2))
 
    if len(tubes)==0:
        return

    save_batch = os.path.join(save_path,'epoch_'+str(epoch))
    save_image = os.path.join(save_batch,'batch_'+str(i))


    color = (255,0,0)
    thickness = 2
    
    
    input_mask = tubes 
    opacity = 0.7
    
    for batch in range(vis_input.shape[0]):

        input_batch = vis_input[batch]
        seg_map = tubes[batch]

        tube_points = (seg_map == 1).nonzero()
        tube_points = tube_points.data.cpu().numpy()
        
        seg_map_batch = []
        for point in tube_points:
            seg_map_batch.append(point[1])
            seg_map_batch.append(point[0])


        #seg_map_batch = seg_map_batch.cpu().numpy()
        seg_map_batch = [int(round(x)) for x in seg_map_batch]

        save_image_folder = os.path.join(save_image, str(batch))

        if not os.path.exists(save_image_folder):
            os.makedirs(save_image_folder)

        for j in range(input_batch.shape[0]):
            
            input_img = input_batch[j]*255.0
            input_img = np.asarray(input_img, np.float64)

            input_mask = np.zeros((input_img.shape[0], input_img.shape[1],1))

            if len(seg_map_batch)>0:
                input_mask[seg_map_batch[1]:seg_map_batch[3], seg_map_batch[0]:seg_map_batch[2]] = 255.0

            input_mask = np.repeat(input_mask,3,axis=2)
            input_mask = np.asarray(input_mask, np.float64)
            
            input_img = cv2.addWeighted(src1=input_img, alpha=opacity, src2=input_mask, beta=1. - opacity, gamma=0, dtype=-1)
            cv2.imwrite(os.path.join(save_image_folder,f'img_{j}.png' ),input_img)

'''
def weighted_binary_cross_entropy(output,target,weights=None):
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
               
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))
'''


def train(run_id, epoch, train_dataloader, model, optimizer, criterion, criterion_classification, writer, use_cuda, lr_scheduler=None, lr_list= None):

    print('train at epoch {}'.format(epoch))
   
    total_losses = []
   
    model.train()
    start_time = time.time()
    
    if lr_list is not None:
        index = epoch%len(lr_list)
        if epoch>0 and index==0:
            lr_list.reverse()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_list[index]

    for param_group in optimizer.param_groups:
        print('lr: ',param_group['lr'],flush=True)
    #f_w.write("train epoch: %s\n"%(str(epoch)))
    for i, (inputs, tubes, targets, label_masks, _) in enumerate(train_dataloader):
        
        inputs = torch.stack(inputs,dim=0)
        tubes = torch.stack(tubes,dim=0)        
        targets = torch.stack(targets,dim=0)
        label_masks = torch.stack(label_masks,dim=0)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            label_masks = label_masks.cuda()
            tubes = tubes.cuda()
            model.cuda()
            criterion_classification.cuda()
            #criterion.cuda()

        
       
        #print('targets: ',targets)
        optimizer.zero_grad()
        #print('tubes from dl: ', tubes.size())
        #print('targets: ',targets)
        class_label = model(inputs, tubes)
        #print('targets: ',targets.size())
        #print('class_label: ',class_label.size())

        total_loss = criterion_classification( class_label, targets )
        
        total_loss.backward()
        optimizer.step()
        
        

        total_losses.append(total_loss.item())
       
        
        if i % 30 == 0:
            
            print(f'Training Epoch {epoch}, Batch {i}::: Total Loss:{np.mean(total_losses)} ')
            #visualize(inputs, tubes, run_id, epoch, i)
            #targets = targets.data.cpu().numpy()
            #f_w.write("batch: %s\n"%(str(i)))
            #f_w.write("targets: %s\n"%str(targets))
            #f_w.flush()
            

            '''
            for param_group in optimizer.param_groups:
                print('step: ',i,' - lr: ',param_group['lr'],flush=True)
            '''
        del total_loss, class_label, label_masks, targets, tubes, inputs        #imp for batch fit
        ## this is very important to fit more batch size
    
    #logger.save()
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if epoch % params.checkpoint == 0:
        save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
        states = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    time_taken = time.time() - start_time

    sys.stdout.flush()

    print(f'Training Epoch {epoch}::: Loss: {np.mean(total_losses)}')

    writer.add_scalar('Training Total Loss', np.mean(total_losses), epoch)

    return model



def validation(run_id, epoch, valid_dataloader, model, criterion, criterion_classification, writer, use_cuda, lr_scheduler=None):
    
    print('validation at epoch {}'.format(epoch))
    total_losses = []
    predictions, ground_truth = [], []
    model.eval()
    start_time = time.time()
    #f_w.write("valid epoch: %s\n"%(str(epoch)))
    with torch.no_grad():
        for i, (inputs, tubes, targets, label_masks, _) in enumerate(valid_dataloader):


            inputs = torch.stack(inputs,dim=0)
            tubes = torch.stack(tubes,dim=0)        
            targets = torch.stack(targets,dim=0)
            label_masks = torch.stack(label_masks,dim=0)
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                label_masks = label_masks.cuda()
                tubes = tubes.cuda()
                model.cuda()
                criterion_classification.cuda()


            class_label = model(inputs, tubes)
     
            total_loss =  criterion_classification( class_label, targets )
            total_losses.append(total_loss.item())
            
            targets = targets.cpu().numpy()
            ground_truth.extend(targets)
            predictions.extend(class_label.cpu().data.numpy())


            if i % 30 == 0:            
                print(f'Validation Epoch {epoch}, Batch {i}::: Total Loss:{np.mean(total_losses)}')
                #visualize(inputs, tubes,run_id,epoch,i, valid = True)
                #f_w.write("batch: %s\n"%(str(i)))
                #f_w.write("targets: %s\n"%str(targets))
                #f_w.flush()

            del total_loss, label_masks, tubes, inputs              #imp for batch fit
               
        
        #logger.save()
        
        time_taken = time.time() - start_time
        ground_truth = np.array(ground_truth)
        predictions = np.array(predictions)
        predictions = (np.array(predictions) > params.f1_threshold).astype(int)
        ground_truth = (np.array(ground_truth) > params.f1_threshold).astype(int)
        results_actions = precision_recall_fscore_support(np.array(ground_truth), np.array(predictions), average=None)
        support, f1_scores, cls_precision, recall = results_actions[3], results_actions[2], results_actions[0], results_actions[1]

        print('Validation Epoch: %d, F1-Score: %s' % (epoch, str(f1_scores))) 
        print('Validation Epoch: %d, Cls Precision: %s' % (epoch, str(cls_precision)))
        print('Validation Epoch: %d, Recall: %s' % (epoch, str(recall)))
        print('Validation Epoch: %d, support: %s' % (epoch, str(support)))

        sys.stdout.flush()
        print(f'Validation Epoch {epoch}::: Loss: {np.mean(total_losses)}, F1:{np.mean(f1_scores)}, class precision:{np.mean(cls_precision)}, Recall:{np.mean(recall)}, Time: {time_taken}')


        writer.add_scalar('Validation Total Loss', np.mean(total_losses), epoch)
        writer.add_scalar('Validation F1-Score', np.mean(f1_scores), epoch)
        writer.add_scalar('Validation classification Precision', np.mean(cls_precision), epoch)
        writer.add_scalar('Validation Recall', np.mean(recall), epoch)



class PLM(nn.Module):
    def __init__(self, reduction='none'):
        super(PLM, self).__init__()
        
        self.bce = nn.BCELoss(reduction='none')
        
        self.reduction = reduction

    def forward(self, gt, pred, mask):
        bce = self.bce(gt, pred)
        
        masked_bce = bce*mask
        
        if self.reduction == 'sum':
            return torch.mean(torch.sum(masked_bce, dim=-1))
        elif self.reduction == 'mean':
            return torch.mean(torch.sum(masked_bce, dim=-1)/(torch.sum(mask, dim=-1)+1e-7))
        else:
            return masked_bce
    

def iou_precision_numpy(outputs, targets, threshold=0.2):
    SMOOTH = 1e-6
    outputs[outputs>threshold]=1
    outputs[outputs<1]=0
    intersection = np.sum(np.multiply(outputs, targets))
    precision = np.divide(intersection + SMOOTH, np.sum(targets) + SMOOTH)
    union = (targets + outputs)
    np.putmask(union, union > 0, 1)
    union = np.sum(union)
    iou = (intersection + SMOOTH)/(union + SMOOTH)
    if np.sum(targets) == 0 and np.sum(outputs) == 0:
        iou = 1.0
        precision = 1.0
    return iou, precision


def adjust_lr(optimizer, epoch, init_lr, total_epoch):
    lr = init_lr * (0.00005 ** (epoch // total_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train_locator( run_id, use_cuda ):

    torch.backends.cudnn.enabled=False

    writer = SummaryWriter(os.path.join(cfg.tf_logs_dir, str(run_id)))

    print("Run ID : " + run_id)

    print("Parameters used : ")
    print("batch_size from file: " + str(params.batch_size))
    print("lr: " + str(params.learning_rate))
    print("skip_frames: " + str(params.skip_frames))
    print("frames_per_clip: " + str(params.frames_per_clip))
    print("num_samples: " + str(params.num_samples))

    print("Preprocessing Training Data")

    model = InceptionI3d(num_classes=157, in_channels=3)
    #model_2 = I3D(num_classes = 34)
    saved_model_file = "/home/c3-0/praveen/VIRAT/trained_models/i3d_model_rgb_charades.pt"
    #saved_model_file_2 = "/home/c3-0/mahfuz/MEVA_results/models/03-19-20_1816/model_40.pth"

    
    if saved_model_file is not None:
        #model = load_model(model,saved_model_file)
        model.load_state_dict(torch.load(saved_model_file),strict=False)
        #model_2.load_state_dict(torch.load(saved_model_file_2),strict=False)

        print('model loaded from:', saved_model_file,flush=True)

   
    #exit()
    model.replace_logits(params.num_classes )

    
    #exit()
    if use_cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
            model.cuda()
            # criterion.cuda()

        else:
            print('only one gpu is being used')
            model.cuda()
            # criterion.cuda()

    if params.optim == 'SGD':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=1e-6)

    elif params.optim == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    
    saved_previous_file = "/home/c3-0/mahfuz/Elbit_results/models/06-03-20_0320/model_31.pth"#'/home/c3-0/mahfuz/Elbit_results/models/05-25-20_1432/model_7.pth'
    if saved_previous_file is not None:
        model.load_state_dict(torch.load(saved_previous_file)['state_dict'])
        optimizer.load_state_dict(torch.load(saved_previous_file)['optimizer'])
        print('model loaded from:', saved_previous_file,flush=True)
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params.lr_scaling, patience=3, verbose=False, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 2e-3, 2e-5, step_size_up = 2*len(train_dataloader), step_size_down= 2*len(train_dataloader), mode='exp_range' )
    criterion = SegmentationLosses(cuda=use_cuda).build_loss(mode='dice')
    #criterion_classification = PLM(reduction='sum')
    criterion_classification = nn.BCELoss()


    learning_rate_list = list(np.linspace(1e-3, 1e-5, num=20))

    for epoch in range(32,params.num_epochs):


        train_dataset = MEVADataGenerator('train', params.train_percent, params.train_scales, use_localization_alone = False, use_groundtruth_alone = False)
        print("Number of training samples : " + str(len(train_dataset)),flush=True)
        print('train distribution: ',train_dataset.class_statistics,flush=True)
        print('train ratio: ',train_dataset.ratio[0])
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size_local, shuffle=True, collate_fn=filter_none, num_workers=params.num_workers)
        print('train dataloader: ',len(train_dataloader),flush=True)        
        
        
        model = train(run_id, epoch, train_dataloader, model, optimizer, criterion, criterion_classification, writer, use_cuda, lr_scheduler=None , lr_list = None)

                
        validation_dataset = MEVADataGenerator('validation',params.validation_percent, params.validation_scales, use_localization_alone=False, use_groundtruth_alone=False)
        print("Number of validation samples : " + str(len(validation_dataset)),flush=True)
        print('val distribution: ',validation_dataset.class_statistics,flush=True)
        print('validation ratio: ',validation_dataset.ratio[0])


        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_local, shuffle=True, collate_fn=filter_none, num_workers=params.num_workers)
        print('valid dataloader: ',len(validation_dataloader),flush=True)


        validation(run_id, epoch, validation_dataloader, model, criterion, criterion_classification, writer, use_cuda, lr_scheduler=None)

        
if __name__ == "__main__":
    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    use_cuda = torch.cuda.is_available()
    print('USE_CUDA: ',use_cuda,flush=True)
    train_locator(run_started, use_cuda)
    

