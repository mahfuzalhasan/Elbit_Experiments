import pickle
import os


cache = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72'
validation_list_file = os.path.join(cache,'validation_list.pkl')
train_list_file = os.path.join(cache,'train_list.pkl')
annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/corrected_annotations' 

'''
validation_list = ['vlc-record-2019-08-20-08h05m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-20-05h42m30s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt', 'vlc-record-2019-08-20-06h44m02s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-20-08h11m27s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-20-06h33m43s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-21-16h36m39s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-20-06h43m58s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt','vlc-record-2019-08-20-05h42m26s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt','vlc-record-2019-08-20-08h11m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt','vlc-record-2019-08-20-07h55m41s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt','vlc-record-2019-08-21-19h26m34s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt']
'''
'''
validation_list = ['vlc-record-2019-08-20-08h05m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h43m58s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt',
'vlc-record-2019-08-20-08h11m27s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt',
'vlc-record-2019-08-20-06h44m02s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-21-16h57m11s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt']      #including class running

color_list = ['vlc-record-2019-08-20-08h05m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-20-08h11m27s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt','vlc-record-2019-08-20-06h44m02s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile1-_gt.txt']

gray_list = ['vlc-record-2019-08-20-06h43m58s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt', 'vlc-record-2019-08-20-08h11m24s-rtsp_10.248.32.69_RTP-Multicast_pMediaProfile2-_gt.txt']

pickle.dump(gray_list, open(validation_list_file, 'wb'))
'''
validation_list = pickle.load(open(validation_list_file, 'rb'))
print('validation: ',len(validation_list))

train_list = []
for annot_file in os.listdir(annotation_base_dir):
    if annot_file not in validation_list:
        train_list.append(annot_file)

pickle.dump(train_list, open(train_list_file, 'wb'))
print('train: ',len(train_list))
#pickle.dump(validation_list, open(validation_list_file, 'wb'))


