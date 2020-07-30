
import numpy as np
import os
import re


from operator import itemgetter
from itertools import *
import copy
import pickle


def preprocess_data(data_paths):
    
	processed_data = []
	for key in data_paths.keys():
		data = data_paths[key]
		print(key,': ',len(data))
		processed_data.extend(data)

	return processed_data


def calculate_video_distribution(data):  
	annot_file = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72_color'      
	video_frame_dict = {}
	bbox_frame_dict = {}

	instance_count = 0
	for video_info in data:
		
		video_path = video_info['video_path']
		annotation_path = video_info['annotation_path']
		action = video_info['action_name']
		instance_id = video_info['instane_id']

		if video_path not in video_frame_dict.keys():
		    video_frame_dict[video_path] = {}
		
		frame_info = video_info['frame_bbox']

		for f_n, bboxes in frame_info.items():
			if f_n not in video_frame_dict[video_path]:
				video_frame_dict[video_path][f_n] = {}
			'''
			if f_n not in bbox_frame_dict[video_path].keys():
				bbox_frame_dict[video_path][f_n] = {}
			'''
			if instance_id not in video_frame_dict[video_path][f_n]:
				video_frame_dict[video_path][f_n][instance_id] = []

			video_frame_dict[video_path][f_n][instance_id].extend(bboxes)
			'''
			video_frame_dict[video_path][f_n][action_id]['action_name'] = action

			if actor_id not in video_frame_dict[video_path][f_n][action_id]:
				video_frame_dict[video_path][f_n][action_id][actor_id] = []
			video_frame_dict[video_path][f_n][action_id][actor_id].extend(bboxes)
			'''
		#print('video frame dict: ',video_frame_dict)
		#exit()

	#print(video_frame_dict['/home/c3-0/mahfuz/data/Elbit/Annotated_data/frames/vlc-record-2019-08-20-08h33m22s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-'])
	#exit()
	annot_file = os.path.join( annot_file ,'elbit_video_annots_train.pkl')
	pickle.dump(video_frame_dict, open(annot_file, 'wb'))
	#st = '/home/c3-0/mahfuz/data/Elbit/Annotated_data/frames/vlc-record-2019-08-20-08h33m22s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-'
	return video_frame_dict


if __name__ == '__main__':


	data_file = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72_color/elbit_action_instances_train.pkl'

	class_wise_data = pickle.load(open(data_file, 'rb'))
	print('no of instances: ',len(class_wise_data))

	data = preprocess_data(class_wise_data)
	#frame_wise_distribution = calculate_frame_wise_distribution(data)
	video_distribution = calculate_video_distribution(data)


	'''
	print('video_level_action_bbox: ',len(video_level_action_bbox))
	annot_file = os.path.join(cache_folder, data_split + '_vid_annot.pkl')
	print('dumping file')
	pickle.dump(video_level_action_bbox, open(annot_file, 'wb'))
	'''
