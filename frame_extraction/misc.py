import os
import configuration as cfg


def get_video_lists():
    train_videos = [line.rstrip() for line in open(cfg.train_videos_file).readlines()[1:-1]]
    val_videos = [line.rstrip() for line in open(cfg.val_videos_file).readlines()[1:-1]]
    test_videos = [line.rstrip() for line in open(cfg.test_videos_file).readlines()[1:-1]]
    return train_videos, val_videos, test_videos


def rename_folders():
    videos_extracted = os.listdir(cfg.MEVA_frames_folder)
    for folder_name in videos_extracted:
         renamed_folder = folder_name.replace('_', '.')
         os.rename(os.path.join(cfg.MEVA_frames_folder, folder_name), os.path.join(cfg.MEVA_frames_folder, renamed_folder))


def get_missing_videos():
    train_videos, val_videos, test_videos = get_video_lists()
    videos_list = train_videos + val_videos + test_videos
    MEVA_videos = []
    for video in videos_list:
        if 'VIRAT' not in video:
            MEVA_videos.append(video)
    videos_extracted = os.listdir(cfg.MEVA_frames_folder)
    missing_videos = [video for video in MEVA_videos if video not in videos_extracted]
    videos_not_used = [video for video in videos_extracted if video not in MEVA_videos] 
    return missing_videos   


if __name__ == '__main__':
    #missing_videos = get_missing_videos()
    #print(len(missing_videos))
    train_videos, val_videos, test_videos = get_video_lists()
    video_list = train_videos + val_videos + test_videos
    print(len([video for video in video_list if 'VIRAT' not in video]))
