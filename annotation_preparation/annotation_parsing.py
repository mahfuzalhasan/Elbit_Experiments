import numpy as np
import os
import re


from operator import itemgetter
from itertools import *
import copy
import pickle

'''
annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit/Annotated_data/Annotations/GT'
frame_dir = '/home/c3-0/mahfuz/data/Elbit/Annotated_data/frames'
annotation_files = os.listdir(annotation_base_dir)
annot_file = '/home/c3-0/mahfuz/Elbit/cache'
'''
annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/corrected_annotations'
annotation_files = os.listdir(annotation_base_dir)
frame_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/all_frames'
annot_file = '/home/c3-0/mahfuz/Elbit_05-13-20/cache/videos_72'


video_dict = {}
object_dict = {}
frame_obj_count = {}

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

    for instance, frame_list in obj_frames.items():

        instance_list = separation(frame_list)

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



        if obj in ['standing', 'walking', 'waving', 'long_arm', 'digging']:

            for i_list in instance_list:
                instance_dict = instance_creation(obj, instance, i_list, video)
                annotation_tracker.append(instance_dict)


            
    '''
    if 'long_arm' in current_video_stat.keys():
        print('video: ',video)
        print('instances: ', current_video_stat)
    '''

def instance_creation(action, instance, i_list, video):

    instance_dict = {}
    
    i_list = sorted(i_list)

    annotation_path = os.path.join(annotation_base_dir, video)
    video_name = video[:video.rindex('_')]
    video_path = os.path.join(frame_dir, video_name)

    instance_dict['annotation_path'] = annotation_path
    instance_dict['video_path'] = video_path
    instance_dict['action_name'] = action
    instance_dict['instane_id'] = instance
    instance_dict['start_frame'] = i_list[0]
    instance_dict['end_frame'] = i_list[len(i_list)-1]

    return instance_dict

annotation_validation = ['vlc-record-2019-08-20-08h05m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h43m58s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m27s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h44m02s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt']




for annotation in annotation_files: 
    

    if annotation in annotation_validation:
        print('annotation: ',annotation)
        continue

    obj_frames = {}
    file_path = os.path.join( annotation_base_dir, annotation )
    if annotation not in video_dict:
        video_dict[annotation] = {}

    f = open(file_path,'r')
    count = 0


    for line in f:
        line = line.split(',')
        try:
            frame_no = int(line[0])
            obj_no = int(line[1])
            line_obj = line[1:]
        except:
            print('line: ',line)

    
        if obj_no not in frame_obj_count:
            frame_obj_count[obj_no] = 0


        frame_obj_count[obj_no] += 1


        for i in range(obj_no):
            obj = line_obj[(i*5)+5]
            obj = obj.strip()
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

            instance = obj                    ###walking0/walking1
            check = hasnumbers(obj)
            if check:
                obj = obj[:len(obj)-1]        ####walking0 --> walking


            if instance not in obj_frames:
                obj_frames[instance] = []

            
            obj_frames[instance].append(frame_no)
            '''
            object_dict[obj] += 1
            video_dict[annotation][obj] += 1

            if obj not in obj_video:
                obj_video[obj] = []

            if annotation not in obj_video[obj]:
                obj_video[obj].append(annotation)
           '''
 
        count += 1

    instance_count(obj_frames, annotation)

print('instance counter: ',instance_counter_dict)
print('total annotations: ',len(annotation_tracker))

annot_file = os.path.join( annot_file ,'elbit_instances_train.pkl')
#pickle.dump(annotation_tracker, open(annot_file, 'wb'))



        
