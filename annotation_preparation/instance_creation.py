import numpy as np
import os
import re


from operator import itemgetter
from itertools import *
import copy
import pickle

frame_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/all_frames'
annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/all_annotations'
cache = '/home/c3-0/mahfuz/Elbit_05-13-20/cache/old_validation'

train_list_file = os.path.join(cache,'train_list.pkl')
train_list = pickle.load(open(train_list_file, 'rb'))


validation_list_file = os.path.join(cache,'validation_list.pkl')
validation_list = pickle.load(open(validation_list_file, 'rb'))

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
    for instance_id, frame_dict in obj_frames.items():
        frame_list = list(frame_dict.keys())
        instance_list = separation(frame_list)

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

'''
annotation_validation = ['vlc-record-2019-08-20-08h05m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h43m58s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m27s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h44m02s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt']
'''

def get_instance_id_bbox(frame_id_dict, bbox, p_f_n, f_n):
    previous_instance_ids = frame_id_dict[p_f_n].keys()
    current_instance_ids = frame_id_dict[f_n].keys()
    previous_obj_ids = [iid[:len(iid)-1] for iid in previous_instance_ids]
    current_obj_ids = [iid[:len(iid)-1] for iid in current_instance_ids]
    current_obj_ids_set = set(current_obj_ids)


    for c_o_id in current_obj_ids_set:
        occurence_previous = previous_obj_ids.count(c_o_id)
        if occurence_previous == 0:
            continue
        occurence_present = current_obj_ids.count(c_o_id)
        if occurence_present == occurence_previous:
            continue
        if occurence_previous > occurence_present:          #present 1 , previous 2
            min_dis = 1000000
            min_iid = ''
            for iid in current_instance_ids:
                if c_o_id in iid:
                   bbox = frame_id_dict[f_n][iid]
                for piid, pbbox in frame_id_dict[p_f_n].items():
                    if c_o_id in piid:
                        x1_dis = pbbox[0] - bbox[0]
                        if x1_dis < 0:
                            x1_dis = (-1)*x1_dis
                        if x1_dis < min_dis:
                            min_dis = x1_dis
                            min_iid = piid
                if len(min_iid) > 1:
                    if min_iid not in frame_id_dict[f_n]:
                        frame_id_dict[f_n][min_iid] = []
                    frame_id_dict[f_n][min_iid] = bbox 

 
        elif occurence_present > occurence_previous:          #present 2 , previous 1
            min_dis = 1000000
            min_iid = ''
            distance = {}
            used = {}
            previous_iid = ''
            for iid in current_instance_ids:
                if c_o_id in iid:
                    used[iid] = False
                    bbox = frame_id_dict[f_n][iid]
                for piid, pbbox in frame_id_dict[p_f_n].items():
                    
                    if c_o_id in piid:
                        previous_iid = piid
                        x1_dis = pbbox[0] - bbox[0]
                        if x1_dis < 0:
                            x1_dis = (-1)*x1_dis
                        distance[iid] = x1_dis

            
            min = 10000000
            min_bbox = []
            for iid, dis in distacne.items():                
                if dis<min:
                    min=dis
                    min_iid = previous_iid
                    min_bbox = frame_id_dict[f_n][iid] 
            used[min_iid] = True
            frame_id_dict[f_n][min_iid] = min_bbox

                        
    
    for iid, previous_bbox in frame_id_dict[p_f_n].items():
        o_id = iid[:len(iid)-1]
        if o_id in obj_ids:     #current
            current_bbox = frame_id_dict[f_n][iid]
            xmin_distance = current_bbox[0] - previous_bbox[0]
            if xmin_distance < 0:
                xmin_distance  = (-1)*xmin_distance
            if xmin_distance<min:
                min =  xmin_distance
                min_instance_id =  iid   
    return min_instance_id, bbox
    

for annotation in validation_list: 
    frame_id_dict = {}
    #annotation = 'vlc-record-2019-08-20-06h37m17s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt'
    print('annotation: ',annotation)
    obj_frames = {}
    file_path = os.path.join( annotation_base_dir, annotation )
    f = open(file_path,'r')
    count = 0
    for line in f:
        line = line.split(',')
        frame_no = int(line[0])
        obj_no = int(line[1])
        line_obj = line[1:]
        if frame_no not in frame_id_dict:
            frame_id_dict[frame_no] = {}
        instance_id_list = []
        #print('fn: ',frame_no)
        for i in range(obj_no):
            instance_id_location = (i*5)+5
            ptr = i*5
            obj = line_obj[instance_id_location]
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


            bbox = line_obj[(ptr+1):instance_id_location]
            bbox = [round(int(x)) for x in bbox]
            bbox[2] = bbox[0]+bbox[2]
            bbox[3] = bbox[1]+bbox[3]
            #if instance_id in 
            #instance_id_list.append(instance_id)

            instance_id = obj                    ###walking0/walking1
            obj = obj[:len(obj)-1]                ####walking0 --> walking
            #print('obj: ',obj,' ',instance_id)
            #exit()
            if obj not in ['standing', 'walking', 'long_arm', 'digging', 'waving']:
                continue

            if instance_id in frame_id_dict[frame_no]:
                identity = int(instance_id[len(instance_id)-1])
                identity += 1
                instance_id = obj+str(identity)

            if instance_id not in frame_id_dict[frame_no]:
                frame_id_dict[frame_no][instance_id] = []
            frame_id_dict[frame_no][instance_id] = bbox        
                 
            if instance_id not in obj_frames:
                obj_frames[instance_id] = {}
            if frame_no not in obj_frames[instance_id]:
                obj_frames[instance_id][frame_no] = []
            obj_frames[instance_id][frame_no] = bbox
            #print('ins id: ',instance_id) 
        #frame_entry = frame_id_dict[502]
    count += 1
    instance_count(obj_frames, annotation)

print('instance counter: ',instance_counter_dict)
print('total annotations: ',len(annotation_tracker))
print('total videos: ',count)
#exit()
annot_file = os.path.join( cache ,'elbit_instances_validation.pkl')
pickle.dump(annotation_tracker, open(annot_file, 'wb'))



        
