import pickle
import os


cache = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72'
validation_list_file = os.path.join(cache,'validation_list.pkl')
train_list_file = os.path.join(cache,'train_list.pkl')
annotation_base_dir = '/home/c3-0/mahfuz/data/Elbit_05-09-20/corrected_annotations' 

data = pickle.load(open(train_list_file, 'rb'))
previous_train_data = []
for name in data:
    name = name[:name.rfind('_')]
    previous_train_data.append(name)
print('previous train: ',len(previous_train_data))
file_1 = open('./gray_video.txt', 'r') 
colored_train_data = []
lines = file_1.readlines() 
for line in lines:
    line = line.strip()
    line = line[:line.rfind('.')]
    colored_train_data.append(line)
print('colored data: ',len(colored_train_data))

save_path = '/home/c3-0/mahfuz/Elbit_curriculum_learning/cache/videos_72_gray'
colored_train_list_file = os.path.join(save_path,'train_list.pkl')
train_data = []
for data in colored_train_data:
    if data in previous_train_data:
        train_data.append(data)
    else:
        print('not in previous train: ',data)
        
print('train: ',len(train_data))
pickle.dump(train_data, open(colored_train_list_file, 'wb'))
