import pickle
import os


cache = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72'
validation_list_file = os.path.join(cache,'validation_list.pkl')
train_list_file = os.path.join(cache,'train_list.pkl')
annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/corrected_annotations' 

validation_list = pickle.load(open(validation_list_file, 'rb'))
print('validation: ',len(validation_list))

train_list = []
for annot_file in os.listdir(annotation_base_dir):
    if annot_file not in validation_list:
        train_list.append(annot_file)

pickle.dump(train_list, open(train_list_file, 'wb'))
print('train: ',len(train_list))


