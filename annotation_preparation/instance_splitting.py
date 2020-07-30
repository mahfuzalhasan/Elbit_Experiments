import numpy as np
import os
import re


from operator import itemgetter
from itertools import *
import copy
import pickle


def selecting_frame_dict(frame_dict, start_frame, end_frame):

    frame_list = list(range(start_frame, end_frame+1))
    f_dict = {k: v for k, v in frame_dict.items() if k in frame_list}
    return f_dict    

data_file = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72/elbit_instances_train.pkl'
data = pickle.load(open(data_file, 'rb'))
counter = {}
action_instances = {}

list_1 = []

for video_info in data:

    start_frame = video_info['start_frame']
    end_frame = video_info['end_frame']
    frame_range = end_frame - start_frame + 1

    action = video_info['action_name']
    
    if 'digging' in action or 'waving' in action or 'running' in action:
        r_s = 64
        limit = r_s
    else:
        r_s = 288
        limit = 96
        
    '''
    elif 'running' in action:
        r_s = 16
        limit = r_s
    '''
    
    

    if action not in action_instances:
        action_instances[action] = []

    if frame_range < r_s:        #######ignoring instances<16
        continue
    
    i = start_frame

    while i < end_frame:

        if i+r_s-1 > end_frame:
            break
        instance = {}
        instance['annotation_path'] = video_info['annotation_path']
        instance['video_path'] = video_info['video_path']
        instance['action_name'] = video_info['action_name']
        instance['instane_id'] = video_info['instane_id']
        instance['start_frame'] = i
        instance['end_frame'] = i+(r_s-1)  
        f_dict = selecting_frame_dict(video_info['frame_bbox'], i, i+(r_s-1))
        instance['frame_bbox'] = f_dict     
        action_instances[action].append(instance)
        i += r_s


    if end_frame - i >= limit:
        instance = {}
        instance['annotation_path'] = video_info['annotation_path']
        instance['video_path'] = video_info['video_path']
        instance['action_name'] = video_info['action_name']
        instance['instane_id'] = video_info['instane_id']
        instance['start_frame'] = i
        instance['end_frame'] = end_frame
        f_dict = selecting_frame_dict(video_info['frame_bbox'], i, end_frame)
        instance['frame_bbox'] = f_dict     
        action_instances[action].append(instance)

for k,v in action_instances.items():
    print(k,'    ',len(v))

annot_file = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72'
annot_file = os.path.join( annot_file ,'elbit_action_instances_train.pkl')
#pickle.dump(action_instances, open(annot_file, 'wb'))
