import os
import subprocess
import glob
import multiprocessing
import sys
import cv2
#import configuration as cfg
#from misc import *
import pickle

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def build_video_list(video_files, save_dir):

    already_saved = os.listdir(save_dir)
    process = []
    #video_ids = [video_id[:video_id.rindex('.')] for video_id in video_files]
    for video in video_files:
        #print('video: ',video)
        video_id = video[:video.rindex('.')]        #except .avi/.mp4
        if video_id not in already_saved:
            process.append(video)                   # with .avi/.mp4
    return process

def extract_video_frames(video_file, root_dir, save_dir):
    
    video_path = os.path.join(root_dir, video_file)
    save_folder = video_file[:video_file.rindex('.')]           # except .avi/.mp4
    save_path = os.path.join(save_dir, save_folder)             #folder without .avi/.mp4
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    extracted_frames = len(os.listdir(save_path))
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if os.path.exists(save_path) and extracted_frames == num_frames:
        return
    else:
        print(save_path)
        subprocess.call(['ffmpeg', '-i', video_path, os.path.join(save_path, '%05d.png'), '-hide_banner'])


def check_frames(video_files):
    #root_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/videos/color'
    root_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/21Aug_PM_FreeForm'
    save_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/all_frames'


    incomplete_videos = []
    for i, video_file in enumerate(video_files):

        video_path = os.path.join(root_dir, video_file)
        save_folder = video_file[:video_file.rindex('.')]
        save_path = os.path.join(save_dir, save_folder)
        if not os.path.exists(save_path):
            incomplete_videos.append(video_file)
            continue
        extracted_frames = len(os.listdir(save_path))
        print('video file: ',video_file)
        #print('extracted frames: ',extracted_frames)
        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #print('total frames: ',num_frames)
        #print("checking video : {}".format(i))

        if extracted_frames != num_frames:
            print("Video : {}, Extracted Frames : {}, Total Frames : {}".format(video_file, extracted_frames, num_frames))
            if extracted_frames < num_frames:
                incomplete_videos.append(video_file)

    return incomplete_videos


if __name__ == '__main__':

    #check_frames(video_files)




    index = int(sys.argv[1])

    root_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/21Aug_PM_FreeForm'
    save_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/all_frames'
    #available_annot_path = '/home/c3-0/mahfuz/data/Elbit_05-09-20/new/GT/21Aug_PM_FreeForm_GT'
    #new_annot = [annot[:annot.rindex('_')] for annot in os.listdir(available_annot_path)]
    #print(new_annot)
    
    #video_ids = [video[:video.rindex('.')] for video in video_files if video_id[:video_id.rindex('.')] not in saved_files]

    '''
    video_files = os.listdir(root_dir)
    video_files = [video for video in video_files if 'txt' not in video]        # with .avi/.mp4
    annot_exist = [video for video in video_files if video[:video.rindex('.')] in new_annot]
    print('annot exist: ',annot_exist)
    '''
    #exit()
    video_files = ['vlc-record-2019-08-21-19h02m49s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-.mp4','vlc-record-2019-08-21-09h35m40s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-.mp4','vlc-record-2019-08-21-10h20m47s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-.mp4']
    
    '''
    saved_files = os.listdir(save_dir)
    print('saved videos: ',len(saved_files))
    
    incomplete_videos = check_frames(annot_exist)
    print('incomplete videos: ',incomplete_videos)
    print('no. of incomplete videos :',len(incomplete_videos))
    exit()
    '''
    splits = chunks(video_files, 2)
    splits = [split for split in splits]
    split = splits[index]
    jobs = []
    for param in split:
	    process = multiprocessing.Process(target=extract_video_frames, args=(param, root_dir, save_dir))
	    jobs.append(process)
    for j in jobs:
	    j.start()
    for j in jobs:
	    j.join()

