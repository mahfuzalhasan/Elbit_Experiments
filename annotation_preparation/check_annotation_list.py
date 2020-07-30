import os
import pickle










if __name__ == '__main__':

    cache_folder= '/home/c3-0/mahfuz/Elbit_05-13-20/cache'
    #old_annot_path = '/home/c3-0/mahfuz/data/Elbit/Annotated_data/Annotations/GT'
    #new_annot_path = '/home/c3-0/mahfuz/data/Elbit_05-09-20/all_annotations'
    nw_annot_path = '/home/c3-0/mahfuz/data/Elbit_05-09-20/new/GT/20Aug_noon%20-%20set2_GT'
    #new_video_path = '/home/c3-0/mahfuz/data/Elbit_05-09-20/20Aug_noon'
    #new_video_path = '/home/c3-0/mahfuz/data/Elbit_05-09-20/21Aug_PM_FreeForm'
    new_video_path = '/home/c3-0/mahfuz/data/Elbit_05-09-20/new/5_21'
    corrected_annot_path = '/home/c3-0/mahfuz/data/Elbit_05-09-20/corrected_annotations'

    new_annot = [annot[:annot.rindex('_')] for annot in os.listdir(new_annot_path)]
    #print('annotations: ',new_annot)
    #c_annot = [annot for annot in os.listdir(corrected_annot_path)]
    #uncovered_ones = [x for x in new_annot if x not in c_annot]
    new_videos = [video[:video.rindex('_')] for video in os.listdir(new_video_path)]
    
    for video in new_videos:
        if video not in new_annot:
            print(video)
            
    exit()

    split = ['train', 'validation']
    for data_split in split:
        file_path = os.path.join(cache_folder,data_split+'_list.pkl')
        data = pickle.load(open(file_path, 'rb'))

        print(data_split, ': ',len(data))
        common = list(set(c_annot).intersection(data))
        print('common elements: ',len(common))
        if len(common)<len(data):
            uncovered_ones = [x for x in data if x not in common]
            print('uncovered ones: ',uncovered_ones)
            
            
        
        
    '''
    print('total corrected: ',len(c_annot))


    print('common with new: ',len(common_with_new))

    print('uncovered_ones ones: ', uncovered_ones)
    print('len inc ones: ',len(uncovered_ones))

    #old_annot = [annot for annot in os.listdir(old_annot_path)]

    #common_with_old = list(set(c_annot).intersection(old_annot))
    #common_between_previous = list(set(new_annot).intersection(old_annot))
    #print('common between previous: ',len(common_between_previous))
    common_with_new = list(set(c_annot).intersection(new_annot))
    #print('common with old: ',len(common_with_old))
    '''









