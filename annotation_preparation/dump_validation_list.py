import pickle
import os


cache = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72'
validation_list_file = os.path.join(cache,'validation_list.pkl')
#train_list_file = os.path.join(cache,'train_list.pkl')
annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/corrected_annotations' 

validation_list = ['vlc-record-2019-08-20-08h05m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h43m58s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m27s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h44m02s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt']
pickle.dump(validation_list, open(validation_list_file, 'wb'))

