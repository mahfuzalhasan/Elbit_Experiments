import numpy as np
import os
import re


from operator import itemgetter
from itertools import *
import copy
import pickle

split = ['train','validation']
for data_split in split:
    print('data_split: ',data_split)
    data_file = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72/'+data_split+'_list.pkl'
    #data_file = '/home/c3-0/mahfuz/Elbit/cache/elbit_action_instances_'+data_split+'.pkl'
    #data_file = '/home/c3-0/mahfuz/Elbit_05-13-20/cache/videos_72/elbit_action_instances_'+data_split+'.pkl'
    #data_file = '/home/c3-0/mahfuz/Elbit_05-13-20/cache/videos_72_running_activity/elbit_action_instances_'+data_split+'.pkl'
    #data_file = '/home/c3-0/mahfuz/Elbit_05-13-20/cache/old_validation/elbit_action_instances_'+data_split+'.pkl'
    #data_file = '/home/c3-0/mahfuz/Elbit/cache/elbit_video_annots_bbox_train.pkl'
    data = pickle.load(open(data_file, 'rb'))
    print(data_split,' data: ',len(data))

    '''
    count = 0
    print('data file: ',data_file)
    for k,v in data.items():
	    print('action: ',k, ' instances: ',len(v))
    '''

