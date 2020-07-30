import numpy as np
import os
import re


from operator import itemgetter
from itertools import *
import copy
import pickle


annotation_validation = ['vlc-record-2019-08-20-08h05m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h43m58s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m27s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h44m02s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt']


data_file = '/home/c3-0/mahfuz/Elbit_05-13-20/cache/validation_list.pkl'

data = pickle.load(open(data_file, 'rb'))

print(len(data))

print(data)

for path in data:
    if path in annotation_validation:   
        print('path: ',path)
