import numpy as np
import os
import re


from operator import itemgetter
from itertools import *
import copy
import pickle

annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/corrected_annotations'
frame_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/all_frames'
annot_file = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72_attention'
annotation_files = os.listdir(annotation_base_dir)


train_list_file = os.path.join(annot_file,'train_list.pkl')
train_list = pickle.load(open(train_list_file, 'rb'))
print('train file: ',len(train_list))
'''
validation_list_file = os.path.join(annot_file,'validation_list.pkl')
validation_list = pickle.load(open(validation_list_file, 'rb'))
print('validation list: ',validation_list)
'''

obj_video = {}
false_list = {}

instance_counter_dict = {}

annotation_tracker = []

def hasnumbers(input_string):
    return bool(re.search(r'\d', input_string))


def separation(frame_list):
    list_1 = []
    for k, g in groupby(enumerate(frame_list), lambda x:x[1]-x[0]):
        list_1.append(list(map(itemgetter(1), g)))
    return list_1



def instance_count(obj_frames, video):
    
    current_video_stat = {}
    #instance_dict = {}

    for instance_id, frame_dict in obj_frames.items():

        frame_list = list(frame_dict.keys())
        #print('frame list: ',frame_list)
        instance_list = separation(frame_list)
        #print('instance_list: ',instance_list)


        check = hasnumbers(instance_id)

        if check:
            obj = instance_id[:len(instance_id)-1]
        else:
            obj = instance_id


        if obj not in instance_counter_dict:
            instance_counter_dict[obj] = 0
        if obj not in current_video_stat:
            current_video_stat[obj] = 0
        current_video_stat[obj] += len(instance_list)
        if obj in ['standing', 'walking', 'long_arm', 'digging', 'waving']:
            instance_counter_dict[obj] += len(instance_list)
            length = len(instance_list)
            for i_list in instance_list:
                instance_dict = instance_creation(obj, instance_id, frame_dict, length, i_list, video)
                annotation_tracker.append(instance_dict)
    return current_video_stat

def instance_creation(action, instance_id, frame_dict, length, i_list, video):

    instance_dict = {}

    i_list = sorted(i_list)
    start_frame = i_list[0]
    end_frame = i_list[len(i_list)-1]
    frame_list = list(range(start_frame, end_frame+1))
    #print('length: ',length)


    f_dict = {k: v for k, v in frame_dict.items() if k in frame_list}
    #print('frame range: ',frame_dict[start_frame:end_frame+1])
    '''
    if length>1:
        #print('frame dict: ', frame_dict)
        #print('f_dict: ',f_dict)
        print('len fdict: ',len(f_dict.keys()))
        print('len frame list: ',len(frame_list))
        #exit()
    '''
    annotation_path = os.path.join(annotation_base_dir, video)
    video_name = video[:video.rindex('_')]
    video_path = os.path.join(frame_dir, video_name)

    instance_dict['annotation_path'] = annotation_path
    instance_dict['video_path'] = video_path
    instance_dict['action_name'] = action
    instance_dict['instane_id'] = instance_id
    instance_dict['frame_bbox'] = f_dict
    instance_dict['start_frame'] = start_frame
    instance_dict['end_frame'] = end_frame
    
 
    return instance_dict



for annotation in train_list: 
    #print('annotation: ',annotation)
    obj_frames = {}
    #annotation = annotation+'_gt.txt'
    file_path = os.path.join( annotation_base_dir, annotation )

    f = open(file_path,'r')
    count = 0


    for line in f:
        line = line.split(',')
        #print('line: ',line)
        try:
            frame_no = int(float(line[0]))
            obj_no = int(line[1])
            line_obj = line[1:]
        except:
            print('annot: ',annotation)
            print('line: ',line)

        for i in range(obj_no):
            instance_id_location = (i*5)+5
            ptr = i*5
            #print('instance id location: ',instance_id_location)
            try:
                
                obj = line_obj[instance_id_location]
            except:
                print('line: ',line)
                print(instance_id_location)
            obj = obj.strip()
            bbox = line_obj[(ptr+1):instance_id_location]
            bbox = [int(x) for x in bbox]
            bbox[2] = bbox[0]+bbox[2]
            bbox[3] = bbox[1]+bbox[3]
            #print('bbox: ',bbox)

            ######checking for annotation correctness#####################
            if obj.isdigit():
                if annotation not in false_list:
                    false_list[annotation] = {}
                    false_list[annotation]['frame_number'] = []
                    false_list[annotation]['action'] = []
                   
                false_list[annotation]['frame_number'].append(frame_no)
                false_list[annotation]['action'].append(obj)
                continue
            ######checking for annotation correctness#################

            instance_id = obj                    ###walking0/walking1
            check = hasnumbers(obj)
            if check:
                obj = obj[:len(obj)-1]        ####walking0 --> walking

            if obj not in ['standing', 'walking', 'long_arm', 'digging', 'waving']:
                continue

            if instance_id not in obj_frames:
                obj_frames[instance_id] = {}
            if frame_no not in obj_frames[instance_id]:
                obj_frames[instance_id][frame_no] = []
            obj_frames[instance_id][frame_no] = bbox
        
        count += 1
    current_video_stat = instance_count(obj_frames, annotation)
    print('current video stat: ',current_video_stat)

print('instance counter: ',instance_counter_dict)
print('total annotations: ',len(annotation_tracker))
#exit()
annot_file = os.path.join( annot_file ,'elbit_instances_train.pkl')
pickle.dump(annotation_tracker, open(annot_file, 'wb'))



        
