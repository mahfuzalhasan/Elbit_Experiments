import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
#import videotransforms
import torch
import torchvision.utils as vutil
import torch.nn.functional as F


#from i3d import InceptionI3d
from i3dpt import I3D
from model_partial_class import I3D_partial
from loss import *
#from dataloader_loc_feat_pooling import *
from dataloader_loc_feature_pooling_augmentation import *
from custom_loss import WeightedTwoPartBCELoss
import sys
import math

from detectron_2_check_initial import get_model

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
from utils.array_util import *

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from skimage.measure import label, regionprops

#logger = HistorySaver('_results/logs/i3d.npz')
batch_size_local = 8


def get_dists(y_gt_onehot, y_pred_probs):
    # y_gt_onehot is the ground-truth labels of shape (N, C)
    # y_pred_probs is the predicted probabilities of shape (N, C)
    n_classes = params.num_classes
    def kl_div(p, q):
        return np.sum(p*np.log2(p/q))
    def js_div(p, q):
        return 0.5*kl_div(p, 0.5*(p+q))+0.5*kl_div(q, 0.5*(p+q))
    n_bins = 4
    hist_pos_gt = np.zeros((n_bins,)) + 1e-7
    hist_pos_gt[-1] = 1
    hist_pos_gt = hist_pos_gt / np.sum(hist_pos_gt)
    hist_neg_gt = np.zeros((n_bins,)) + 1e-7
    hist_neg_gt[0] = 1
    hist_neg_gt = hist_neg_gt / np.sum(hist_neg_gt)
    d_pos, d_neg = np.zeros((n_classes, )), np.zeros((n_classes, ))
    for i in range(n_classes):
        pred_probs = y_pred_probs[:, i]
        gt_probs = y_gt_onehot[:, i]
        hist_pos_pred = np.histogram(pred_probs[gt_probs == 1], bins=n_bins, range=(0, 1))[0] + 1e-7
        hist_pos_pred = hist_pos_pred/np.sum(hist_pos_pred)
        hist_neg_pred = np.histogram(pred_probs[gt_probs == 0], bins=n_bins, range=(0, 1))[0] + 1e-7
        hist_neg_pred = hist_neg_pred / np.sum(hist_neg_pred)
        d_pos[i] = js_div(hist_pos_pred, hist_pos_gt)
        d_neg[i] = js_div(hist_neg_pred, hist_neg_gt)
    return d_pos, d_neg

def gt_tube_coordinates(tube_map):
    tube = []
    tube_points = (tube_map == 1).nonzero()
    tube_points = tube_points.data.cpu().numpy()
    ###print('tube points: ',tube_points)

    for point in tube_points:
        tube.append(point[1])
        tube.append(point[0])

    return tube

def bbox_merge(bboxes):
    
    bbox = [min([box[0] for box in bboxes]), min([box[1] for box in bboxes]), max([box[2] for box in bboxes]), max([box[3] for box in bboxes])] #x_min,y_min,x_max,y_max
    return bbox

def reshape(tube, height, width, expected_height = params.frames_input_height, expected_width = params.frames_input_width):
    new_tube = []

    ratio_h = expected_height/height
    ratio_w = expected_width/width

    new_tube = [tube[0] * ratio_w, tube[1]*ratio_h, tube[2] * ratio_w, tube[3]*ratio_h]
    ##print('new tube: ',new_tube)

    return new_tube
    

def visualize(inputs, tubes, run_id, epoch, i, valid = False):
    input_sv_shape = inputs.permute(0,2,1,3,4).shape
    vis_input = inputs.permute(0,2,1,3,4)
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



def get_bbox_map(batch, check, inputs, model):
    outputs = []
    inputs = inputs.cpu().numpy()
    #print('inputs shape: ',inputs.shape)
    inputs = inputs[0]
    inputs = np.transpose(inputs,(1,2,3,0))
    for index in range(inputs.shape[0]):
        output = np.zeros((448,800), np.float32)
        frame = inputs[index]*255.0
        #cv2.imwrite('input.jpg',frame)
        #print('frame shape: ',frame.shape)
        predictions = model(frame)
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy() if predictions["instances"].has("pred_boxes") else None
        #print('pred bbox: ',boxes)
        if boxes is not None:
            if check:
                print('batch ', batch, ' det boxes: ',boxes)
            for bbox in boxes:
                bbox = [int(round(x)) for x in bbox]
                output[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
        #cv2.imwrite('map.jpg',output*255)
        outputs.append(output)   
        #exit()        
    outputs = np.asarray(outputs)
    #print('outputs shape: ',outputs.shape)   
    return outputs  
        


def validation(run_id, epoch, valid_dataloader, model, model_classification, loc_model, criterion, criterion_classification, writer, use_cuda, lr_scheduler=None):
    
    print('validation at epoch {}'.format(epoch),flush=True)
    total_losses = []
    predictions_1, predictions_2, predictions_3, ground_truth = [], [], [], []
    #model.eval()
    model_classification.eval()
    start_time = time.time()
    #f_w.write("valid epoch: %s\n"%(str(epoch)))


    save_path = params.local_output_dir
    save_path = os.path.join(save_path, run_id)
    save_path = os.path.join(save_path, 'valid')


    with torch.no_grad():
        for i, (inputs, tubes, targets, label_masks, possible_targets) in enumerate(valid_dataloader):
        
            print_order = False
            
            if i%20 == 0:
                print_order = True
        


            inputs = torch.stack(inputs,dim=0)
            tubes = torch.stack(tubes,dim=0)        
            targets = torch.stack(targets,dim=0)
            label_masks = torch.stack(label_masks,dim=0)
            possible_targets = torch.stack(possible_targets,dim=0)

            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                label_masks = label_masks.cuda()
                tubes = tubes.cuda()
                possible_targets = possible_targets.cuda()
                #model.cuda()
                #model_classification.cuda()
                criterion_classification.cuda()
                #criterion.cuda()

            _, _, _, _, mixed_3c = loc_model(inputs)
            outputs_numpy_b = get_bbox_map(i, print_order, inputs, model)  #f,h,w --> 16,448,800
            ########################################################localization tube extraction

            labels = []
            tube_map = tubes[0]               
            target = targets[0]
            inputs_b = inputs[0]
            #############output drawn    
            if i%1==0:
                save_epoch = os.path.join(save_path,str(epoch))
                save_batch = os.path.join(save_epoch,str(i))
                save_batch = os.path.join(save_batch,'loc_mask')
                if not os.path.exists(save_batch):
                    os.makedirs(save_batch)
                for k in range(outputs_numpy_b.shape[0]):
                    mask_batch = outputs_numpy_b[k]
                    mask_batch = np.expand_dims(mask_batch,axis=2)
                    #resized = cv2.resize(mask_batch, (800,448), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(save_batch,f'valid_loc_mask_itr_{i}_clip_0_frame_{k}.png' ),mask_batch*255)
                inputs_b = inputs_b.cpu().data.numpy()
                inputs_b = np.transpose(inputs_b, (1, 2, 3, 0)) 

                for k in range(inputs_b.shape[0]):
                    resized = inputs_b[k]
                    cv2.imwrite(os.path.join(save_batch,f'valid_input_itr_{i}_clip_0_frame_{k}.png' ),resized*255)
            ##########################   
            bboxes = []
            lbl = label(outputs_numpy_b)
            blobs = regionprops(lbl)
            for blob in blobs:                          #each blob.bbox is calculated over 16 frames.. 
                if blob.area<500:
                    continue
                bbox_yx = blob.bbox          ##format y1,x1,y2,x2
                #print('batch ',i, ' blob bbox: ',bbox_yx)
                bbox = [bbox_yx[2],bbox_yx[1],bbox_yx[5],bbox_yx[4]]    ##format x1,y1,x2,y2
                if print_order:
                    print('batch ',i, ' blob bbox_xy: ',bbox)
                #bbox = reshape(bbox,224,400)                            ##224,400 --> 448,800
                bbox = check_min_criteria( bbox, params.frames_input_height, params.frames_input_width )
                if print_order:
                    print('batch ',i, ' blob bbox_xy after splating: ',bbox)
                bboxes.append(bbox)                                     

            ######multi forward pass
            generated_label_1 = []
            generated_label_2 = []
            generated_label_3 = []
            
            if len(bboxes)>0:
                for l_tube in bboxes: 
                    localization_tube_mask = np.zeros((outputs_numpy_b.shape[1], outputs_numpy_b.shape[2]))

                    loc_tube_coords = [int(math.floor(x)) for x in l_tube]
                    if loc_tube_coords[0]<0:
                        loc_tube_coords[0] = 0 
                    if loc_tube_coords[1]<0:
                        loc_tube_coords[1] = 0 
                    if loc_tube_coords[2]>=params.frames_input_width:
                        loc_tube_coords[2] = params.frames_input_width-1 
                    if loc_tube_coords[3]>=params.frames_input_height:
                        loc_tube_coords[3] = params.frames_input_height-1 

              
                    localization_tube_mask[loc_tube_coords[1],loc_tube_coords[0]] = 1                        #ymin   #xmin
                    localization_tube_mask[loc_tube_coords[3],loc_tube_coords[2]] = 1                         #ymax   xmax
                    localization_tube_mask = np.asarray(localization_tube_mask, dtype='f')
                    localization_tube_mask = torch.from_numpy(localization_tube_mask)
                    localization_tube_mask = torch.unsqueeze(localization_tube_mask,dim=0)

                    class_label_b = model_classification(mixed_3c, localization_tube_mask)
                   
                    t2 = torch.tensor([params.label_threshold]).cuda()
                    class_label_b_2 =  (class_label_b > t2).float() * 1

                    t3 = torch.tensor([0.3]).cuda()
                    class_label_b_3 =  (class_label_b > t3).float() * 1

                    t1 = torch.tensor([0.1]).cuda()
                    class_label_b_1 =  (class_label_b > t1).float() * 1

                    generated_label_1.append(class_label_b_1)
                    generated_label_2.append(class_label_b_2)
                    generated_label_3.append(class_label_b_3)
              
            else:  
                localization_tube_mask = np.zeros((outputs_numpy_b.shape[1], outputs_numpy_b.shape[2])) 
                localization_tube_mask = np.asarray(localization_tube_mask, dtype='f')
                localization_tube_mask = torch.from_numpy(localization_tube_mask)
                class_label_b = model_classification(mixed_3c, localization_tube_mask)

                t2 = torch.tensor([params.label_threshold]).cuda()
                class_label_b_2 =  (class_label_b > t2).float() * 1

                t3 = torch.tensor([0.3]).cuda()
                class_label_b_3 =  (class_label_b > t3).float() * 1

                t1 = torch.tensor([0.1]).cuda()
                class_label_b_1 =  (class_label_b > t1).float() * 1

                generated_label_1.append(class_label_b_1)
                generated_label_2.append(class_label_b_2)
                generated_label_3.append(class_label_b_3)
                 
               
                
                
            class_label_1 = torch.zeros(generated_label_1[0].size()).cuda()
            for c_label in generated_label_1:
                class_label_1 += c_label

            class_label_2 = torch.zeros(generated_label_2[0].size()).cuda()
            for c_label in generated_label_2:
                class_label_2 += c_label

            class_label_3 = torch.zeros(generated_label_3[0].size()).cuda()
            for c_label in generated_label_3:
                class_label_3 += c_label

            ##print('class label after addition: ',class_label)
            class_label_1[class_label_1>0] = 1
            class_label_1 = class_label_1.cuda()

             
            class_label_2[class_label_2>0] = 1
            class_label_2 = class_label_2.cuda()

             
            class_label_3[class_label_3>0] = 1
            class_label_3 = class_label_3.cuda()
            
            ###########################################################localization tube extraction done

            ##print('targets: ',possible_targets)
            #print('size targets: ',possible_targets.size())


            
            total_loss = criterion_classification.forward( class_label_1, possible_targets )
            total_loss = torch.mean(total_loss)

            total_losses.append( total_loss.item() )
            
            possible_targets = possible_targets.cpu().numpy()
            ground_truth.extend(possible_targets)
            predictions_1.extend(class_label_1.cpu().data.numpy())
            predictions_2.extend(class_label_2.cpu().data.numpy())
            predictions_3.extend(class_label_3.cpu().data.numpy())

            
            if i % 40 == 0:       
                print(f'Validation Epoch {epoch}, Batch {i}::: Total Loss:{np.mean(total_losses)}',flush=True)
                '''
                visualize(inputs, tubes,run_id,epoch,i, valid = True)
                f_w.write("batch: %s\n"%(str(i)))
                f_w.write("targets: %s\n"%str(targets))
                f_w.flush()

                save_epoch = os.path.join(save_path,str(epoch))
                save_batch = os.path.join(save_epoch,str(i))
                save_batch = os.path.join(save_batch,'loc_mask')
            
                if not os.path.exists(save_batch):
                    os.makedirs(save_batch)
                
                #################output draw
                output_sv_shape = outputs.permute(0,2,1,3,4).shape
                vis_output = outputs.permute(0,2,1,3,4).contiguous().view(-1, output_sv_shape[2], output_sv_shape[3], output_sv_shape[4])
                vutil.save_image(vis_output,os.path.join(save_batch,f'torch_loc_mask_{epoch}_batch_{i}.png'), range=(0.0,1.0), normalize=True, nrow=4)
                '''
            del total_loss, label_masks, tubes, inputs              #imp for batch fit
               
        
        #logger.save()
        
        time_taken = time.time() - start_time
        ground_truth = np.array(ground_truth)
        ground_truth = (np.array(ground_truth) > params.f1_threshold).astype(int)

        ### the way label_threshold has been handled, this f1_threshold is of no significance here
        predictions_1 = np.array(predictions_1)
        predictions_1 = (np.array(predictions_1) > params.f1_threshold).astype(int)

        predictions_2 = np.array(predictions_2)
        predictions_2 = (np.array(predictions_2) > params.f1_threshold).astype(int)

        predictions_3 = np.array(predictions_3)
        predictions_3 = (np.array(predictions_3) > params.f1_threshold).astype(int)

        results_actions = precision_recall_fscore_support(np.array(ground_truth), np.array(predictions_1), average=None)
        support, f1_scores_1, cls_precision_1, recall_1 = results_actions[3], results_actions[2], results_actions[0], results_actions[1]

        print('Validation Epoch: %d, support: %s' % (epoch, str(support)),flush=True)

        print('th 0.1')
        print('Validation Epoch: %d, F1-Score: %s' % (epoch, str(f1_scores_1)),flush=True) 
        print('Validation Epoch: %d, Cls Precision: %s' % (epoch, str(cls_precision_1)),flush=True)
        print('Validation Epoch: %d, Recall: %s' % (epoch, str(recall_1)),flush=True)
        

        results_actions = precision_recall_fscore_support(np.array(ground_truth), np.array(predictions_2), average=None)
        _, f1_scores_2, cls_precision_2, recall_2 = results_actions[3], results_actions[2], results_actions[0], results_actions[1]

        print('th 0.2',flush=True)
        print('Validation Epoch: %d, F1-Score: %s' % (epoch, str(f1_scores_2)),flush=True) 
        print('Validation Epoch: %d, Cls Precision: %s' % (epoch, str(cls_precision_2)),flush=True)
        print('Validation Epoch: %d, Recall: %s' % (epoch, str(recall_2)),flush=True)

       

        results_actions = precision_recall_fscore_support(np.array(ground_truth), np.array(predictions_3), average=None)
        _, f1_scores_3, cls_precision_3, recall_3 = results_actions[3], results_actions[2], results_actions[0], results_actions[1]

        print('th 0.3',flush=True)
        print('Validation Epoch: %d, F1-Score: %s' % (epoch, str(f1_scores_3)),flush=True) 
        print('Validation Epoch: %d, Cls Precision: %s' % (epoch, str(cls_precision_3)),flush=True)
        print('Validation Epoch: %d, Recall: %s' % (epoch, str(recall_3)),flush=True)
         
        sys.stdout.flush()
        print(f'Validation Epoch {epoch}::: Loss: {np.mean(total_losses)}, F1_0.1:{np.mean(f1_scores_1)}, F1_0.2:{np.mean(f1_scores_2)}, F1_0.3:{np.mean(f1_scores_3)}, class precision_0.2:{np.mean(cls_precision_2)}, Recall_0.2:{np.mean(recall_2)}, Time: {time_taken}',flush=True)



        writer.add_scalar('Validation Total Loss_0.1', np.mean(total_losses), epoch)
        writer.add_scalar('Validation F1-Score_0.1', np.mean(f1_scores_1), epoch)
        writer.add_scalar('Validation classification Precision_0.1', np.mean(cls_precision_1), epoch)
        writer.add_scalar('Validation Recall_0.1', np.mean(recall_1), epoch)

        writer.add_scalar('Validation F1-Score_0.2', np.mean(f1_scores_2), epoch)
        writer.add_scalar('Validation classification Precision_0.2', np.mean(cls_precision_2), epoch)
        writer.add_scalar('Validation Recall_0.2', np.mean(recall_2), epoch)

        writer.add_scalar('Validation F1-Score_0.3', np.mean(f1_scores_3), epoch)
        writer.add_scalar('Validation classification Precision_0.3', np.mean(cls_precision_3), epoch)
        writer.add_scalar('Validation Recall_0.3', np.mean(recall_3), epoch)



class PLM(nn.Module):
    def __init__(self, reduction='none'):
        super(PLM, self).__init__()
        
        self.bce = nn.BCELoss(reduction='none')
        
        self.reduction = reduction

    def forward(self, gt, pred, mask):
        bce = self.bce(pred, gt)
        
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

def weight_distribution(train_dataset):
    distribution = train_dataset.class_statistics
    print('distribution: ',distribution)
    total_data = len(train_dataset) - distribution[len(distribution)-1]
    print('total_data: ',total_data)
    pos_weights = np.zeros(len(distribution))
    neg_weights = np.zeros(len(distribution))
    
    for k,v in distribution.items():            
        if k<len(distribution) - 1:
            if v>0:
                pos_weights[k] = total_data / v
                neg_weights[k] = total_data / (total_data - v)
            else:
                pos_weights[k] = 0
                neg_weights[k] = 0
            
    sum_pos_weights = 0
    sum_neg_weights = 0    
    for i in range(len(pos_weights)):
        sum_pos_weights += pos_weights[i]
        sum_neg_weights += neg_weights[i]
    for i in range(len(pos_weights)):
        pos_weights[i] = pos_weights[i]/sum_pos_weights
        neg_weights[i] = neg_weights[i]/sum_neg_weights
        
    pos_weights[len(distribution)-1] = 0.1
    neg_weights[len(distribution)-1] = 0.1
    
    pos_weights = torch.from_numpy(pos_weights)
    neg_weights = torch.from_numpy(neg_weights)
    
    return pos_weights, neg_weights


def train_locator( run_id, use_cuda ):

    torch.backends.cudnn.enabled=False

    writer = SummaryWriter(os.path.join(cfg.tf_logs_dir, str(run_id)))

    print("Run ID : " + run_id,flush=True)

    print("Parameters used : ",flush=True)
    print("batch_size from file: " + str(params.batch_size),flush=True)
    print("lr: " + str(params.learning_rate),flush=True)
    print("Preprocessing Training Data")

    loc_model = I3D()
    model = get_model()
    model_classification = I3D_partial(num_classes = 36)
    saved_model_file = "/home/c3-0/mahfuz/MEVA_results/models/04-07-20_2121/model_27.pth"
    
    
    if saved_model_file is not None:
        model_classification.load_state_dict(torch.load(saved_model_file)['state_dict_cls'])
        print('model loaded from:', saved_model_file,flush=True)
    model_classification.replace_logits(params.num_classes)
    
    
    if use_cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            loc_model = nn.DataParallel(loc_model)
            loc_model.cuda()
            model_classification = nn.DataParallel(model_classification)
            model_classification.cuda()
        else:
            print('only one gpu is being used')
            model_classification.cuda()
            loc_model.cuda()
            
            
    pretrained_loc_model = '/home/c3-0/mahfuz/Elbit_results/models/06-02-20_2015/model_24.pth'
    if pretrained_loc_model is not None:
        loc_model.load_state_dict(torch.load(pretrained_loc_model)['state_dict'])
        print('loc model loaded from: ',pretrained_loc_model, flush=True)
    for p in loc_model.parameters():
        p.requires_grad = False


    
    if params.optim == 'SGD':
        print("Using SGD optimizer",flush=True)
        optimizer = torch.optim.SGD(model_classification.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=1e-6)
    elif params.optim == 'ADAM':
        print("Using ADAM optimizer",flush=True)
        optimizer = torch.optim.Adam(model_classification.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)    
        
        

        
    classification_model_path = '/home/c3-0/mahfuz/Elbit_results/models/06-05-20_0003/model_30.pth'
    if classification_model_path is not None:
        model_classification.load_state_dict(torch.load(classification_model_path)['state_dict_cls'])
        print('cls model loaded from: ',classification_model_path, flush=True)
        optimizer.load_state_dict(torch.load(classification_model_path)['optimizer'])
        
    


    
    criterion = SegmentationLosses(cuda=use_cuda).build_loss(mode='dice')
    criterion_classification = nn.BCELoss()
    learning_rate_list = list(np.linspace(1e-3, 1e-5, num=20))
    
    

    
    for epoch in range(params.num_epochs):
        validation_dataset = MEVADataGenerator('validation', params.validation_percent, params.validation_scales, use_localization_alone=False, use_groundtruth_alone=False)
        print("Number of validation samples : " + str(len(validation_dataset)),flush=True)
        print('validation ratio: ',validation_dataset.ratio[0])
        pos_weights, neg_weights = weight_distribution(validation_dataset)
        print('pos weights: ',pos_weights)
        print('neg weights: ',neg_weights)
        criterion_classification = WeightedTwoPartBCELoss(use_cuda, pos_weights, neg_weights)
        validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, collate_fn=filter_none, pin_memory=True, num_workers=4)
        print('valid dataloader: ',len(validation_dataloader),flush=True)
        validation(run_id, epoch, validation_dataloader, model, model_classification, loc_model, criterion, criterion_classification, writer, use_cuda, lr_scheduler=None)
       

            
if __name__ == "__main__":
    run_started = datetime.today().strftime('%m-%d-%y_%H%M')
    use_cuda = torch.cuda.is_available()
    print('USE_CUDA: ',use_cuda,flush=True)
    train_locator(run_started, use_cuda)
    

