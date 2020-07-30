import numpy as np
import os
import re

from operator import itemgetter
from itertools import *
import copy


color_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/videos/color'
gray_base_dir =  '/home/c3-0/mahfuz/data/Elbit_05-09-20/videos/gray'

annotation_files_color = [files for files in os.listdir(color_base_dir) if 'txt' in files]
annotation_files_gray = [files for files in os.listdir(gray_base_dir) if 'txt' in files]



video_dict = {}
object_dict = {}
frame_obj_count = {}

obj_video = {}
false_list = {}

instance_counter_dict = {}

def hasnumbers(input_string):
    return bool(re.search(r'\d', input_string))


def separation(frame_list):
    list_1 = []
    for k, g in groupby(enumerate(frame_list), lambda x:x[1]-x[0]):
        list_1.append(list(map(itemgetter(1), g)))

    return list_1

def instance_count(obj_frames, video):
    #print('obj_frames: ',obj_frames)
    current_video_stat = {}
    
    for instance, frame_list in obj_frames.items():
        instance_list = separation(frame_list)
        '''
        C = copy.deepcopy(instance_list)
        C = sorted(set(map(tuple, C)), reverse=True)
        for sub_list in instance_list:
            if len(sub_list)<16:
                
                d = list(set(C) - set(sub_list))
                C = copy.deepcopy(d)
        '''
        
        check = hasnumbers(instance)
        if check:
            obj = instance[:len(instance)-1]
        else:
            obj = instance

        if obj not in instance_counter_dict:
            instance_counter_dict[obj] = 0
        if obj not in current_video_stat:
            current_video_stat[obj] = 0

        current_video_stat[obj] += len(instance_list)
        instance_counter_dict[obj] += len(instance_list)
    #if 'long_arm' in current_video_stat.keys():
    print('video: ',video)
    print('instances: ', current_video_stat)
    #exit()

for annotation in annotation_files_gray: 
    obj_frames = {}
    file_path = os.path.join( gray_base_dir, annotation )
    if annotation not in video_dict:
        video_dict[annotation] = {}

    f = open(file_path,'r')
    count = 0

    for line in f:
        line = line.split(',')
        frame_no = int(line[0])
        obj_no = int(line[1])
        line_obj = line[1:]
        
        if obj_no not in frame_obj_count:
            frame_obj_count[obj_no] = 0

        frame_obj_count[obj_no] += 1
        instance_id_list = []
        for i in range(obj_no):
            try:
                obj = line_obj[(i*5)+5]
            except:
                print('fr no: ',frame_no)
                print('annot: ',annotation)
                exit()
            obj = obj.strip()
            if obj.isdigit():
                if annotation not in false_list:
                    false_list[annotation] = {}
                    false_list[annotation]['frame_number'] = []
                    false_list[annotation]['action'] = []
                   
                false_list[annotation]['frame_number'].append(frame_no)
                false_list[annotation]['action'].append(obj)
                continue
                
                
            check = hasnumbers(obj)
            instance = obj
            if check:
                obj = obj[:len(obj)-1]
            '''
            if obj not in ['standing', 'walking', 'long_arm', 'digging', 'waving']:
                continue
            '''
            if instance in instance_id_list:
                identity = int(instance[len(instance)-1])
                identity += 1
                instance = obj+str(identity)
            instance_id_list.append(instance)
            if obj not in video_dict[annotation]:
                video_dict[annotation][obj] = 0
            if instance not in obj_frames:
                obj_frames[instance] = []

            obj_frames[instance].append(frame_no)
            video_dict[annotation][obj] += 1

            if obj not in obj_video:
                obj_video[obj] = []

            if annotation not in obj_video[obj]:
                obj_video[obj].append(annotation)
                
        count += 1
    instance_count(obj_frames, annotation)
    #break

print('instance counter: ',instance_counter_dict)
#print('video distribution: ', obj_video)
#exit()


        
