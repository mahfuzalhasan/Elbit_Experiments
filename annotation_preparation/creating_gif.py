from PIL import Image, ImageDraw
import numpy as np
import cv2
import random
import pickle
import os


class_counter = {'walking':0,'waving':0,'digging':0,'long_arm':0,'standing':0}
def gif_save(data_file_path):
    images = []
    data = pickle.load(open(data_file_path, 'rb'))

    for video_info in data:

        start_frame = video_info['start_frame']
        end_frame = video_info['end_frame']
        frame_range = end_frame - start_frame + 1
        print('frame range: ',frame_range)
        action = video_info['action_name']
    
        print('action: ',action)
        #continue
        #if frame_range < 120:
        #    continue
        if 'waving' not in action:
            continue
        '''
        else:
            print(video_info['video_path'])
            print('frame range: ',frame_range)
            class_counter[action] += 1
            continue
        '''
        
        class_counter[action] += 1
        if class_counter[action]>3:
            continue
        
        images = []
        if frame_range > 120:
            start_frame += random.randint(0,frame_range-120)
        video_path = video_info['video_path'] 
        print('video_path: ',video_path)
        frame_list = sorted(os.listdir(video_path))

        
        count = 0
        for i in range(start_frame, end_frame):
            frame_path = os.path.join(video_path, frame_list[i])
            frame = Image.open(frame_path)
            images.append(frame)
            count += 1
            if count>120:
                break
        image_name = video_path[video_path.rindex('/')+1:]+'_'+action+'_'+str(class_counter[action])+'.gif'
        image_path = os.path.join('/home/c3-0/mahfuz/Elbit/gif_save',image_name)
        images[0].save(image_path, save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    print('class counter: ',class_counter)

if __name__ == '__main__':
    file_path = '/home/c3-0/mahfuz/Elbit/cache/elbit_instances_bbox_train.pkl'  
    gif_save(file_path)
    
