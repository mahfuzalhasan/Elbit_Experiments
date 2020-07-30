import json
import os
import pickle
import random
import time
import torch
import copy

import configuration as cfg
import cv2
import h5py as h5
import numpy as np
import parameters as params
#from keras.utils import to_categorical
from torch.utils.data import Dataset, DataLoader
from utils.array_util import *
#from annot_util import AnnotationReader
from datetime import datetime

import re

import torchvision.utils as vutil



class MEVADataGenerator(Dataset):

    def __init__(self, data_split, data_percentage, scales, use_localization_alone=False, use_groundtruth_alone=False, shuffle=True, transform=None, ratio = None):

        self.data_split = data_split        
        self.overlap_threshold = params.overlap_threshold if data_split == 'train' else 0.9
        self.classes = json.load(open(cfg.classes_json, 'r'))['classes']
        self.shuffle = shuffle

        self.classwise_data = self.build_paths()
        #self.classwise_data = self.classwise_distribution()
        self.frame_wise_distribution = self.load_video_level_annotation()

        ###########aug tag add        
        self.augmentation_tag()
        ###########pre-process per class

        self.data_overall, self.samples_per_class = self.preprocess_data(self.classwise_data, params.num_samples)
        
        print(len(self.data_overall))
        self.data, labels = self.precomputation()
        len_data = len(self.data)
       
        self.class_statistics = self.calculate_distribution(labels)
        print('self statistics: ',self.class_statistics)

         
        ##################
        self.n_samples_total = len_data
        n_pos = np.array([self.class_statistics[x] for x in range(params.num_classes)])
        ratios = n_pos / (self.n_samples_total - n_pos)
        if ratio is None:
            self.ratio = np.ones((params.num_classes, ))*np.mean(ratios)#0.05#0.15#np.mean(ratios)
        else:
            self.ratio = ratio        
        self.scales = scales
        assert (self.data_split in ['train', 'validation', 'test'])
        if self.data_split == 'train':
            self.augmentation = params.train_augmentation 
        elif data_split == 'validation':
            self.augmentation = params.validation_augmentation
        else:
            self.augmentation = params.test_augmentation
        self.batch_size = params.batch_size
        self.use_localization_alone = use_localization_alone
        self.use_groundtruth_alone = use_groundtruth_alone
        if self.shuffle:
            random.shuffle(self.data)  # shuffling videos
        #self.data = self.data[0:20]
        

    def __len__(self):
        return len(self.data)


    def get_mask(self, y_i, i):

        n_i = self.class_statistics[i]
        n_hat_i = (self.n_samples_total - n_i)
        if n_i/n_hat_i > self.ratio[i] and y_i == 1:
            rand_n = np.random.random_sample()
            return 1 if rand_n <= (n_hat_i*self.ratio[i])/n_i else 0
        elif n_i/n_hat_i < self.ratio[i] and y_i == 0:
            rand_n = np.random.random_sample()
            return 1 if rand_n <= n_i/(n_hat_i*self.ratio[i]) else 0
        else:
            return 1


    def load_video_level_annotation(self):
        data_file = os.path.join(cfg.cache_folder, 'elbit_video_annots_'+self.data_split + '.pkl')
        assert os.path.exists(data_file)
        data = pickle.load(open(data_file, 'rb'))
        return data


    def build_paths(self):
        data_file = os.path.join(cfg.cache_folder, 'elbit_action_instances_' + self.data_split + '.pkl')
        assert os.path.exists(data_file)
        data = pickle.load(open(data_file, 'rb'))
        for k,v in data.items():
            print(k,':',len(v))
        return data


    def classwise_distribution(self):
        class_data_storage = {}
        for video_info in self.data:
            action = video_info['action']
            if action not in class_data_storage:
                class_data_storage[action] = []
            class_data_storage[action].append(video_info)
        return class_data_storage



    def augment(self,data,repeat=6):
        new_data = []
        if repeat == 6:
            aug_list = ['none','flip','rewind','random_crop','flip-random_crop','rewind-random_crop'] 
        elif repeat == 3:
            aug_list = ['none','flip','rewind','random_crop']
        elif repeat == 1:
            aug_list = ['none']
        for k in range(repeat):
            for entry in data: 
                data_info = copy.deepcopy(entry)
                data_info['augment'] = aug_list[k]
                new_data.append(data_info)
        return new_data



    def augmentation_entry(self,data):
        new_data = []
        if len(data) < params.aug_threshold and self.data_split=='train':
            if len(data)<100:
                new_data = self.augment(data,repeat=6)
            else:
                new_data = self.augment(data,repeat=3)
        else:
            new_data = self.augment(data,repeat=1)

        return new_data



    def augmentation_tag(self):
        for action, data in self.classwise_data.items():
            ###print('action: ',action,' action instances: ',len(data))
            new_data = self.augmentation_entry(data)
            random.shuffle(new_data)
            self.classwise_data[action] = new_data



    def preprocess_data(self, data_paths, limit):
        samples_per_class = {}
        processed_data = []
        for key in data_paths.keys():
            data = data_paths[key]
            
            if len(data) > limit:
                if self.shuffle:
                    random.shuffle(data)
                if self.data_split == 'train':          
                    processed_data.extend(data[0:limit])
                    samples_per_class[key] = limit 
                else:                                   # for validation take all irrespective of the data limit
                    processed_data.extend(data)
                    samples_per_class[key] = len(data)
                    
            else:                                       
                if self.shuffle:
                    random.shuffle(data)
                processed_data.extend(data)
                samples_per_class[key] = len(data)

        return processed_data, samples_per_class



    # this will get changed for each epoch
    def calculate_distribution(self,labels):
        classes = json.load(open(cfg.classes_json, 'r'))['classes']
        distribution = {classes[i]:0 for i in classes.keys()}
        keys = [key for key in classes.keys()]
        for label in labels:
            action_classes = np.where(label == 1)[0]
            for action in action_classes:
                distribution[classes[keys[action]]] += 1
        return distribution


    def per_frame_bbox_retrieval(self, video_info, start_frame, end_frame, skip_frame, repeat):
        bboxes = []
        video_dict = self.frame_wise_distribution[video_info['video_path']]
        action_id = video_info['action_id']
        for f in range(start_frame, end_frame, skip_frame):
            bbox = video_dict[f][action_id]['bbox']
            bboxes.append(bbox)
        return bboxes
                      
        
    def precomputation(self):
        data = []
        data_labels = []
        possible_data_labels = []
        for video_info in self.data_overall:
            train_info = {}
            ################################
            video_path = video_info['video_path']
            action = video_info['action_name']
            start_frame = video_info['start_frame']
            end_frame = video_info['end_frame']
            repeat = False
            recheck = True
            no_of_frames = end_frame - start_frame + 1

            if not os.path.exists(video_path):
                print('vp: ',video_path)
                return None, None

            skip_frames = params.skip_frames            #4 inititally
            frames_per_clip = params.frames_per_clip    #16
            len_slice = skip_frames * frames_per_clip   #64    
            if no_of_frames < len_slice:
                skip_frames = 3
                len_slice = skip_frames * frames_per_clip     
            if no_of_frames < len_slice:
                skip_frames = 2
                len_slice = skip_frames * frames_per_clip
            if no_of_frames < len_slice:
                skip_frames = 1
                len_slice = skip_frames * frames_per_clip     

            if no_of_frames < len_slice:
                repeat = True
            else:
                start_frame += random.randint(0, no_of_frames-len_slice)    #randomly select a start frame
                end_frame = start_frame +  len_slice 


            label, tube, possible_label, crop_bbox, overlapes = self.precomputing_label(video_info, start_frame, end_frame, skip_frames, repeat)
            


            label = np.where(label >= 0.25, 1, 0)
            action_classes = np.where(label == 1)[0]
            if len(action_classes) == 0:
                label = np.append(label,[1])
            else:
                label = np.append(label, [0])
                
            # for capturing all possible actions in current clip
            possible_label = np.where(possible_label >= 0.25, 1, 0)
            possible_action_classes = np.where(possible_label == 1)[0]
            if len(possible_action_classes) == 0:
                possible_label = np.append(possible_label,[1])
            else:
                possible_label = np.append(possible_label, [0])

            assert ( len(label) == params.num_classes)
            assert ( len(possible_label) == params.num_classes)

            train_info['video_path'] = video_path
            train_info['action_name'] = action
            train_info['start_frame'] = start_frame
            train_info['end_frame'] = end_frame
            train_info['skip_frame'] = skip_frames
            train_info['repeat'] = repeat                
            train_info['label'] = label
            #train_info['valid_frame_action'] = valid_frame_action
            #train_info['valid_frame_bbox'] = valid_frame_bbox
            train_info['crop_bbox'] = crop_bbox
            train_info['tube'] = tube
            train_info['frame_bbox'] = video_info['frame_bbox']
            train_info['augment'] = video_info['augment']
            train_info['possible_label'] = possible_label
            train_info['overlapes'] = np.asarray(overlapes, dtype='f')

            data_labels.append(label)
            possible_data_labels.append(possible_label)
            data.append(train_info)
            
        return data, data_labels
                                                                

    def __getitem__(self, index):
        
        #try:
        rgb_clip, seg_maps, label, label_mask, possible_label, overlapes = self.get_sample(index)

        #try:

        rgb_clip = torch.from_numpy(rgb_clip)
        seg_maps = torch.from_numpy(seg_maps)
        label = torch.from_numpy(label)
        label_mask = torch.from_numpy(label_mask)
        possible_label = torch.from_numpy(possible_label)
        overlapes = torch.from_numpy(overlapes)
        return rgb_clip, seg_maps, label, label_mask, possible_label
        '''
        except:
            video_info = self.data[index]
            video_path = video_info['video_path']
            #print('exception occurs',flush=True)
            #print('video path: ',video_path,flush=True)
            
            return None, None, None, None
        '''
        


    def get_sample(self, idx):        
        train_info = self.data[idx]
        rgb_clip, tubes, label, label_mask, possible_label, overlapes = self.process_item(train_info)

        #classes = json.load(open(cfg.classes_json, 'r'))['classes']
        #keys = list(self.classes.keys())
        #action_classes = np.where(label == 1)[0]
        #actions = [keys[i] for i in action_classes]
        #print('actions: ',actions)
        seg_maps = np.zeros((rgb_clip.shape[1], rgb_clip.shape[2]))        
        if len(tubes)>0:
            tubes = [int(round(x)) for x in tubes]
            if tubes[0]<0:
                tubes[0] = 0 
            if tubes[1]<0:
                tubes[1] = 0 
            if tubes[2]>=params.frames_input_width:
                tubes[2] = params.frames_input_width-1 
            if tubes[3]>=params.frames_input_height:
                tubes[3] = params.frames_input_height-1

            if tubes[0]>=params.frames_input_width:
                tubes[0] = params.frames_input_width-1

            if tubes[1]>=params.frames_input_height:
                tubes[1] = params.frames_input_height-1
            ################################
            seg_maps[tubes[1],tubes[0]] = 1                         #ymin   #xmin
            seg_maps[tubes[3],tubes[2]] = 1                         #ymax   xmax

        rgb_clip = np.asarray(rgb_clip, dtype='f')
        seg_maps = np.asarray(seg_maps, dtype='f')
        label = np.asarray(label, dtype='f')
        possible_label = np.asarray(possible_label, dtype='f')
        label_mask = np.asarray(label_mask, dtype='f')
        rgb_clip = np.transpose(rgb_clip, (3, 0, 1, 2))
        return rgb_clip, seg_maps, label, label_mask, possible_label, overlapes
        

    def process_item(self, train_info): 
           
        video_path = train_info['video_path']
        action = train_info['action_name']
        start_frame = train_info['start_frame']
        end_frame = train_info['end_frame']
        skip_frame = train_info['skip_frame']
        repeat = train_info['repeat']
        label = train_info['label']
        possible_label = train_info['possible_label']

        if not os.path.exists(video_path):
            return None, None, None, None

        clip, tube, overlapes = self.build_clip( train_info, start_frame, end_frame, skip_frame, repeat)

        augmentation_type = train_info['augment'].split('-')
        #print('tube: ',tube)
        #print('augmentation type: ',augmentation_type)

        if self.data_split == 'train':
            for aug in augmentation_type:
                if aug=='flip':
                    clip = self.augment_clip_flip(clip)
                    tube = self.tube_flip(tube)
                if aug=='rewind':
                    clip = self.augment_clip_rewind(clip)
                 
        assert ( len(clip) == params.frames_per_clip )
        assert ( len(label) == params.num_classes)

        if self.data_split == 'train':
            label_mask = np.array([self.get_mask(label[i], i) for i in range(label.shape[0])])
        else:
            label_mask = np.ones((label.shape[0], ))


        #print('tube returning: ',tube)
        return clip, tube, label, label_mask, possible_label, overlapes


    def augment_clip_flip(self, clip):
        flipped_clip = []
        for i in range(len(clip)):
            flipped_clip.append(np.fliplr(clip[i]))
        flipped_clip = np.array(flipped_clip)
        return flipped_clip

    def tube_flip(self,tube, width = params.frames_input_width):

        if len(tube)==0:
            return np.array(tube)

        new_tube = [width-tube[2], tube[1], width-tube[0], tube[3]]
        new_tube = np.array(new_tube)
        return new_tube
        

    def augment_label_flip(self, label):
        new_label = np.zeros(len(label))
        for index in np.where(label == 1)[0]:
            if index in params.augmentation_mapping['flip'].keys():
                new_label[params.augmentation_mapping['flip'][index]] = 1
            else:
                new_label[index] = 1
        return new_label



    def augment_clip_rewind(self, clip):
        rewound_clip = np.flip(np.array(clip), axis=0)
        return rewound_clip

    def augment_label_rewind(self, label):
        new_label = np.zeros(len(label))
        for index in np.where(label == 1)[0]:
            if index in params.augmentation_mapping['rewind'].keys():
                new_label[params.augmentation_mapping['rewind'][index]] = 1
            else:
                new_label[index] = 1
        return new_label


    def get_crop_location(self, clip_width, clip_height):
        crop_position = random.randint(0,8)
        crop_height = int(clip_height/1.2)
        crop_width = int(clip_width/1.2)
        x_crop = -1
        y_crop = -1
        #if clip_tag == 'center':
        if crop_position == 0:
            anchor_x, anchor_y = int(clip_width/2), int(clip_height/2)
            x_crop = anchor_x - int(crop_width/2)
            y_crop = anchor_y - int(crop_height/2)
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'left_top':
        if crop_position == 1:
            x_crop, y_crop = 0, 0
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'left':
        if crop_position == 2:
            anchor_x, anchor_y = 0, int(clip_height/2)
            x_crop = 0
            y_crop = anchor_y - int(crop_height/2)
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'left_bottom':
        if crop_position == 3:
            anchor_x, anchor_y = 0, clip_height
            x_crop = 0
            y_crop = anchor_y - crop_height
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'top':
        if crop_position == 4:
            anchor_x, anchor_y = int(clip_width/2), 0
            x_crop = anchor_x - int(crop_width/2)
            y_crop = 0
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'bottom':
        if crop_position == 5:
            anchor_x, anchor_y = int(clip_width/2), clip_height
            x_crop = anchor_x - int(crop_width/2)
            y_crop = anchor_y - crop_height
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'right_top':
        if crop_position == 6:
            anchor_x, anchor_y = clip_width, 0
            x_crop = anchor_x - crop_width
            y_crop = 0
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'right':
        if crop_position == 7:
            anchor_x, anchor_y = clip_width, int(clip_height / 2)
            x_crop = anchor_x - crop_width
            y_crop = anchor_y - int(crop_height / 2)
            #return x_crop, y_crop, crop_width, crop_height

        #if clip_tag == 'right_bottom':
        if crop_position == 8:
           anchor_x, anchor_y = clip_width, clip_height
           x_crop = anchor_x - crop_width
           y_crop = anchor_y - crop_height

        return x_crop, y_crop, crop_width, crop_height
       
    def bbox_merge(self,bboxes):
        #bbox = [int(round(np.min(bboxes[:,0]))), int(round(np.min(bboxes[:,1]))), int(round(np.max(bboxes[:,2]))), int(round(np.max(bboxes[:,3])))]
        bbox = [min([box[0] for box in bboxes]), min([box[1] for box in bboxes]), max([box[2] for box in bboxes]), max([box[3] for box in bboxes])] #x_min,y_min,x_max,y_max
        return bbox


    def get_coordinates(self, reference_bbox, compared_box):
        x_min = max(reference_bbox[0],compared_box[0])
        y_min = max(reference_bbox[1],compared_box[1])
        x_max = min(reference_bbox[2],compared_box[2])
        y_max = min(reference_bbox[3],compared_box[3])
        return [x_min,y_min,x_max,y_max]


    def hasnumbers(self, input_string):
        return bool(re.search(r'\d', input_string))


    def instance_to_action(self, instance_id):
        check = self.hasnumbers(instance_id)
        if check:
            action = instance_id[:len(instance_id)-1]
        else:
            action = instance_id
        return action



    def mapping(self, reference_bbox, displacement):
        reference_bbox[0] = reference_bbox[0] - displacement[0]
        reference_bbox[1] = reference_bbox[1] - displacement[1]
        reference_bbox[2] =  reference_bbox[2] - displacement[0]
        reference_bbox[3] = reference_bbox[3] - displacement[1] 
        return reference_bbox
        
        
        
    def get_all_actions_bboxes(self, video_dict, start_frame, end_frame, skip_frame):

        actions = np.zeros((params.frames_per_clip, params.num_classes-1))
        count = -1
        for f in range(start_frame, end_frame, skip_frame):
            instance_ids = video_dict[f].keys()
            count += 1
            for i_id in instance_ids:
                #bbox = video_dict[f][i_id]        
                #valid_frame_bbox[f].append(bbox)
                action = self.instance_to_action(i_id)  
                if 'digging' in action:
                    continue                                    
                actions[count][self.classes[action]] = 1
                #valid_frame_action[f].append(action)

        return actions



    def precomputing_label(self, video_info, start_frame, end_frame, skip_frame, repeat):
        actions = np.zeros((params.frames_per_clip, len(self.classes)-1))
        video_dict = self.frame_wise_distribution[video_info['video_path']]
        list_frames = sorted(os.listdir(video_info['video_path']))
        frame_path = os.path.join(video_info['video_path'],list_frames[0])
        frame = cv2.imread(frame_path)
        height = frame.shape[0]
        width = frame.shape[1]
        random_crop = False
        # not the case with elbit
        if len(video_dict) == 0:                #images with no actions
            label = np.zeros(len(classes)-1) 
            tube = []
            return label, tube

        augmentation_type = video_info['augment'].split('-')
        for aug in augmentation_type:
            if aug=='random_crop':
                random_crop = True

        reference_bboxes = []  
        tube = []
        displacement = []

        crop_bbox = []
        if random_crop:
            x_crop, y_crop, crop_width, crop_height =  self.get_crop_location(frame.shape[1], frame.shape[0])       
            crop_bbox = [x_crop, y_crop, x_crop + crop_width, y_crop + crop_height]
            displacement = [crop_bbox[0], crop_bbox[1]]
            #print('crop bbox: ',crop_bbox)       

         
        reference_instance_id = video_info['instane_id']  
        invalidity_count_reference_bbox = 0
        count = -1

        possible_actions = self.get_all_actions_bboxes(video_dict, start_frame, end_frame, skip_frame)
        

        for f in range(start_frame, end_frame, skip_frame):
            count += 1  
            try:
                reference_bbox = []
                bbox = video_dict[f][reference_instance_id]
                for x in bbox:
                    if x<0:
                        x = 0
                    reference_bbox.append(x)
                  
            except:
                print(start_frame,'     ',end_frame, '      ',f)
                print('video_info: ',video_info)

            if random_crop:
                
                overlap = get_bbox_overlap(reference_bbox, crop_bbox)
                #print('reference bbox: ',reference_bbox)
                if overlap > self.overlap_threshold and overlap < 1:
                    reference_bbox = self.get_coordinates(reference_bbox, crop_bbox)
                #print('new coordinate: ',reference_bbox)
                reference_bbox = self.mapping(reference_bbox, displacement)
                #print('after mapping: ',reference_bbox)
                reference_bboxes.append(reference_bbox)                                         
                overlap = 1 if overlap > self.overlap_threshold else 0
                if overlap < 1:
                    invalidity_count_reference_bbox += 1
                    continue

            if not random_crop:                                                                 
                reference_bboxes.append(reference_bbox)

            #reference_bboxes.append(reference_bbox)
            reference_action = self.instance_to_action(reference_instance_id)       
            if 'digging' not in reference_action:                                
                actions[count][self.classes[reference_action]] = 1 

        
        tube = self.bbox_merge(reference_bboxes) 
        
        
        if random_crop:
            height = crop_bbox[3] - crop_bbox[1]
            width = crop_bbox[2] - crop_bbox[0]
        
        tube = self.reshape(tube, height, width)
        tube = [int(round(x)) for x in tube]        
        tube = self.check_min_criteria( tube, params.frames_input_height, params.frames_input_width )

        overlapes = []
        count = -1
        for f in range(start_frame, end_frame, skip_frame):
            count += 1
            if f not in video_dict.keys():
                continue
            
            instance_ids_current_frame = video_dict[f].keys()
            for i_id in instance_ids_current_frame:
                if i_id == reference_instance_id:
                    continue

                bbox_other_instance = []
                bbox = video_dict[f][i_id]

                for x in bbox:
                    if x<0:
                        x = 0
                    bbox_other_instance.append(x)

                if random_crop:
                    #print('bbox other: ',bbox_other_instance)
                    overlap = get_bbox_overlap(bbox_other_instance, crop_bbox)
                    overlap = 1 if overlap > self.overlap_threshold else 0
                    if overlap<1:
                        continue
                    bbox_other_instance = self.get_coordinates(bbox_other_instance, crop_bbox)
                    #print('new coordinate other bbox: ',bbox_other_instance)
                    bbox_other_instance = self.mapping(bbox_other_instance, displacement)
                    #print('mapped other bbox: ',bbox_other_instance)

                bbox_other_instance = self.reshape(bbox_other_instance, height, width) 
                overlap = get_bbox_overlap(bbox_other_instance, tube)
                overlap = 1 if overlap > self.overlap_threshold else 0
                overlapes.append(overlap)
                if overlap < 1:
                    continue
                other_action = self.instance_to_action(i_id)   
                if 'digging' not in other_action:                                    
                    actions[count][self.classes[other_action]] = 1 


        if repeat:
            frames_per_clip = params.frames_per_clip
            repeat_amount = frames_per_clip - (count+1)
            pivot = count + 1            
            while repeat_amount > 0:                    # Repeating here
                actions[pivot] = actions[count]         # setting action label for newly repeated last frame
                possible_actions[pivot] = possible_actions[count]
                pivot += 1
                repeat_amount -= 1
                
        label = np.zeros(len(self.classes)-1) 
        possible_label = np.zeros(params.num_classes-1)
        
        for j in range(len(label)):
            label[j] = np.mean(actions[:,j])
            possible_label[j] = np.mean(possible_actions[:,j])

        return label, tube, possible_label, crop_bbox, overlapes



    def reshape(self,tube, height, width, expected_height = params.frames_input_height, expected_width = params.frames_input_width):
        new_tube = []

        ratio_h = expected_height/height
        ratio_w = expected_width/width

        new_tube = [tube[0] * ratio_w, tube[1]*ratio_h, tube[2] * ratio_w, tube[3]*ratio_h]
        #print('new tube: ',new_tube)

        return new_tube

    #############unused currently
    # if tube width< 128 and height<72 then 
    def splating_bbox(self, tube, threshold, frames_input_height, frames_input_width):
        width_tube = tube[2] - tube[0] + 1
        height_tube = tube[3] - tube[1] + 1


        if width_tube < threshold:                      
            width_offset = threshold - width_tube       
            last_half = math.ceil(width_offset/2)
            first_half = math.floor(width_offset/2)
            if tube[2] + last_half < frames_input_width:  
                if tube[0] - first_half >=0:  
                    tube[2] = tube[2] + last_half
                    tube[0] = tube[0] - first_half
                else:
                    tube[2] = tube[2] + width_offset
                    
            else:                                                     
                tube[0] = tube[0] - width_offset

        if height_tube < threshold:
            height_offset = threshold - height_tube
            last_half = math.ceil(height_offset/2)
            first_half = math.floor(height_offset/2)

            if tube[3] + last_half < frames_input_height:
                if tube[1] - first_half >= 0:
                    tube[3] = tube[3] + last_half
                    tube[1] = tube[1] - first_half
                else:
                    tube[3] = tube[3] + height_offset        
            else:
                tube[1] = tube[1] - height_offset

        return tube

    #############unused currently
    # if tube width< 128 and height<72 then 
    # tube_h or tube_w < 112 ....make it 112x112.. if it is >112 say, 200x500... make it square 
    def check_min_criteria( self, tube, frames_input_height, frames_input_width, min_threshold = 96):


        width_tube = tube[2] - tube[0] + 1
        height_tube = tube[3] - tube[1] + 1

        if width_tube < min_threshold and height_tube < min_threshold:
            threshold = min_threshold
            tube = self.splating_bbox(tube, threshold, frames_input_height, frames_input_width)

        elif width_tube < min_threshold or height_tube < min_threshold:
            threshold = max(width_tube, height_tube)
            tube = self.splating_bbox(tube, threshold, frames_input_height, frames_input_width)    
        else:
            threshold = max(width_tube, height_tube)
            tube = self.splating_bbox(tube, threshold, frames_input_height, frames_input_width)
            

        return tube    
    
    #################unused 
    #################unused            
        
    def build_clip(self, train_info, start_frame, end_frame, skip_frame, repeat):
        
        frames = []
        bbox_frames = []
        width = -1
        height = -1
        tube = []  
        
        # get all video number in sorted order from path
        list_frames = sorted(os.listdir(train_info['video_path']))
        
        bbox_counter = 0
        crop_bbox = train_info['crop_bbox']
        count = -1
        overlapes = train_info['overlapes']
        for i in range(start_frame, end_frame, skip_frame):
            frame_path = os.path.join(train_info['video_path'],list_frames[i])
            frame = cv2.imread(frame_path)
            bbox_frame = np.zeros((frame.shape[0],frame.shape[1]))
            height = frame.shape[0]
            width = frame.shape[1]


            if len(crop_bbox) > 0:            
                cropped_region = frame[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
            else:
                cropped_region = frame


            rescaled_region = cv2.resize(cropped_region, (params.frames_input_width, params.frames_input_height))
            rescaled_region = rescaled_region / 255.0            
            frames.append(rescaled_region)
            count += 1
            bbox_counter += 1 # no need here

        if len(train_info['tube'])>0:
            tube = train_info['tube']

        if repeat:
            frames_per_clip = params.frames_per_clip
            cur_element_no = len(frames)
            repeat_amount = frames_per_clip - cur_element_no
            last_frame = frames[len(frames)-1]
            #pivot = count+1
            while repeat_amount > 0:                            # Repeating here
                frames.append(last_frame)
                repeat_amount -= 1
        return np.array(frames), np.array(tube), overlapes


def filter_none(batch):
    rgb_clips, seg_maps, labels, label_masks, overlapes = [], [], [], [], []
    for item in batch:
        if item[0] is not None and item[1] is not None and item[2] is not None and item[3] is not None and item[4] is not None:
            rgb_clips.append(item[0])
            seg_maps.append(item[1])
            labels.append(item[2])
            label_masks.append(item[3])
            overlapes.append(item[4])        
    return rgb_clips, seg_maps, labels, label_masks, overlapes

if __name__ == '__main__':
    shuffle = True
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print('run id: ',run_id)
    #data_generator = MEVADataGenerator('train', 1.0, params.train_scales, use_localization_alone=False, use_groundtruth_alone=False)
    train_dataset = MEVADataGenerator('train', params.train_percent, params.train_scales, use_localization_alone = False, use_groundtruth_alone = False)
    start = time.time()
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=shuffle, num_workers=1, collate_fn=filter_none)
    epoch = 0
    
    opacity = 0.5
    for i, (inputs, seg_maps, targets, label_masks, possible_targets) in enumerate(dataloader):
        
        inputs = torch.stack(inputs,dim=0)
        seg_maps = torch.stack(seg_maps,dim=0)   
        #print('tube from dl: ',seg_maps.size())     
        targets = torch.stack(targets,dim=0)
        print('batch: ',i,' targets: ',targets)

        #overlapes = torch.stack(overlapes,dim=0)
        #print('batch: ',i,' overlapes: ',overlapes)

        label_masks = torch.stack(label_masks,dim=0)
        '''
        if i %2 ==0:
            pickle.dump(inputs, open('train_frames.pkl', 'wb'))
            pickle.dump(targets, open('train_labels.pkl', 'wb'))
            pickle.dump(seg_maps, open('train_seg_maps.pkl', 'wb'))
        '''

        input_sv_shape = inputs.permute(0,2,1,3,4).shape
        print('input sv shape: ',input_sv_shape)
        vis_input = inputs.permute(0,2,1,3,4)#.contiguous().view(-1, input_sv_shape[2], input_sv_shape[3], input_sv_shape[4])
        save_path = os.path.join( params.output_image_save, run_id )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        vis_input = vis_input.data.cpu().numpy()
        vis_input = np.transpose(vis_input,(0,1,3,4,2))
        print('vis input shape: ',vis_input.shape)

        if len(seg_maps)==0:
            continue

        save_batch = os.path.join(save_path,'folder_'+str(i))
        color = (255,0,0)
        thickness = 2
        input_mask = seg_maps         
        for batch in range(vis_input.shape[0]):
            input_batch = vis_input[batch]
            seg_map = seg_maps[batch]
            tube_points = (seg_map == 1).nonzero()
            tube_points = tube_points.data.cpu().numpy()
            seg_map_batch = []
            for point in tube_points:
                seg_map_batch.append(point[1])
                seg_map_batch.append(point[0])

            #print('input batch shape: ',input_batch.shape[0])
            seg_map_batch = [int(round(x)) for x in seg_map_batch]
            save_image = os.path.join(save_batch,'batch_'+str(batch))
            if not os.path.exists(save_image):
                os.makedirs(save_image)

            for j in range(input_batch.shape[0]):                
                input_img = input_batch[j]*255.0
                input_img = np.asarray(input_img, np.float64)
                input_mask = np.zeros((input_img.shape[0],input_img.shape[1],1))
                if len(seg_map_batch)>0:
                    input_mask[seg_map_batch[1]:seg_map_batch[3], seg_map_batch[0]:seg_map_batch[2]] = 255.0
                input_mask = np.repeat(input_mask,3,axis=2)
                input_mask = np.asarray(input_mask, np.float64)
                input_img = cv2.addWeighted(src1=input_img, alpha=opacity, src2=input_mask, beta=1. - opacity, gamma=0, dtype=-1)
                cv2.imwrite(os.path.join(save_image,f'img_{j}.png' ),input_img)
        #if i==10:
        #    exit()
        
    
