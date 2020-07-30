resnet_shortcut = 'B' # possible values A/B
wide_resnet_k = 2
resnext_cardinality = 32
checkpoint = 1
#arch = '{}-{}'.format("i3D", "50")
arch = 'i3D'
input_size = 224
skip_frames = 4
frames_per_clip = 16

allow_frame_repeat = True

optim = 'ADAM' # use 'ADAM' or 'SGD'
learning_rate = 1e-4
learning_rate_2 = 5e-5
dampening = 0
momentum = 0.9
weight_decay = 1e-6
nesterov = True
lr_patience = 10

num_samples = 350
num_samples_loc = 600
num_epochs = 2000

higher_sampling_classes = [0]
medium_sampling_classes = [1,2]
higher_sampling_rate = 350
medium_sampling_rate = 180

batch_size = 6
output_dir = '/home/c3-0/mahfuz/Elbit_results/output'
local_output_dir = '/home/c3-0/mahfuz/Elbit_curriculum_learning/output'
validation_percent = 1.0
train_percent = 1.0
save_frequency = 1

pretrained_loc_model = '/home/c3-0/rizve/diva/DivaExperiments/Localization/experiment_4_multi_Layer_loss_atrous_decoder/trained_models/01-07-19_1745/model_22.pth'#'/home/c3-0/mahfuz/Elbit_results/models/05-27-20_0020/model_16.pth'#'/home/c3-0/mahfuz/Elbit_results/models/05-25-20_0347/model_12.pth'

train_augmentation = ['none', 'flip', 'rewind']
validation_augmentation = ['none']
test_augmentation = ['none']
label_threshold = 0.2

num_workers = 8
saved_models = '/home/c3-0/ishan/saved_models_MEVA/action_classification/kitware5_24-11-19_0503/model_22.pth'
saved_models_dir = '/home/c3-0/mahfuz/Elbit_results/models'
logs_dir = '/home/c3-0/mahfuz/Elbit_results/logs'
#mode = 'start' # 'start' or 'restart', 'start' by default

train_scales = [0.6, 0.7, 0.8]
validation_scales = [0.7]
test_scales = [0.7]

#augmentation_mapping = {'rewind':{0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10, 16: 17, 17: 16, 19: 20, 20: 19, 23: 24, 24: 23, 30: 31, 31: 30, 32: 37, 33: 34, 34: 33}, 'flip':{16: 17, 17: 16}}
#augmentation_mapping = {'rewind':{0: 1, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10, 19: 20, 23: 24, 24: 23, 30: 31, 31: 30, 33: 34, 34: 33}}
#augmentation_mapping = {'rewind':{4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10, 19: 20, 30: 31, 31: 30, 33: 34, 34: 33}}

augmentation_mapping = {'rewind':{4: 10, 10: 4, 1: 7, 7: 1, 8: 11, 11: 8, 12: 14, 14: 12}, 'flip':{}}

use_localization_alone = False
use_groundtruth_alone = False

frames_input_height = 448
frames_input_width = 800

overlap_threshold = 0.7
f1_threshold = 0.2
aug_threshold = 100

curriculum_epoch = 5

num_classes = 6
mode = 'rgb'

output_image_save = '/home/c3-0/mahfuz/Elbit_curriculum_learning/output'
