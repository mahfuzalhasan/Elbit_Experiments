import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict
import copy


class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3d(nn.Module):
	"""Inception-v1 I3D architecture.
	The model is introduced in:
		Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
		Joao Carreira, Andrew Zisserman
		https://arxiv.org/pdf/1705.07750v1.pdf.
	See also the Inception architecture, introduced in:
		Going deeper with convolutions
		Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
		Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
		http://arxiv.org/pdf/1409.4842v1.pdf.
	"""

	# Endpoints of the model in order. During construction, all the endpoints up
	# to a designated `final_endpoint` are returned in a dictionary as the
	# second return value.
	VALID_ENDPOINTS = (
		'Conv3d_1a_7x7',
		'MaxPool3d_2a_3x3',
		'Conv3d_2b_1x1',
		'Conv3d_2c_3x3',
		'MaxPool3d_3a_3x3',
		'Mixed_3b',
		'Mixed_3c',
		'MaxPool3d_4a_3x3',
		'Mixed_4b',
		'Mixed_4c',
		'Mixed_4d',
		'Mixed_4e',
		'Mixed_4f',
		'MaxPool3d_5a_2x2',
		'Mixed_5b',
		'Mixed_5c',
		'Logits',
		'Predictions',
	)

	def __init__(self, num_classes=400, spatial_squeeze=True,
			     final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5, multi_label=True):
		"""Initializes I3D model instance.
		Args:
		  num_classes: The number of outputs in the logit layer (default 400, which
			  matches the Kinetics dataset).
		  spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
			  before returning (default True).
		  final_endpoint: The model contains many possible endpoints.
			  `final_endpoint` specifies the last endpoint for the model to be built
			  up to. In addition to the output at `final_endpoint`, all the outputs
			  at endpoints up to `final_endpoint` will also be returned, in a
			  dictionary. `final_endpoint` must be one of
			  InceptionI3d.VALID_ENDPOINTS (default 'Logits').
		  name: A string (optional). The name of this module.
		Raises:
		  ValueError: if `final_endpoint` is not recognized.
		"""
		self.multi_label = multi_label

		if final_endpoint not in self.VALID_ENDPOINTS:
			raise ValueError('Unknown final endpoint %s' % final_endpoint)

		super(InceptionI3d, self).__init__()
		self._num_classes = num_classes
		self._spatial_squeeze = spatial_squeeze
		self._final_endpoint = final_endpoint
		self.logits = None
	
		self.dropout = nn.Dropout(dropout_keep_prob)

		if self._final_endpoint not in self.VALID_ENDPOINTS:
			raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

		self.end_points = {}
		end_point = 'Conv3d_1a_7x7'
		self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
			                                stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point) # changed the stride, was [2, 2, 2], to process 224 x 224 input
		if self._final_endpoint == end_point: return
	
		end_point = 'MaxPool3d_2a_3x3'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
			                                                 padding=0) # changed the stride, was [1, 2, 2], to process 32 frame inputs
		if self._final_endpoint == end_point: return
	
		end_point = 'Conv3d_2b_1x1'
		self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
			                           name=name+end_point)
		if self._final_endpoint == end_point: return
	
		end_point = 'Conv3d_2c_3x3'
		self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
			                           name=name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'MaxPool3d_3a_3x3'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), 
			                                                 padding=0)
		if self._final_endpoint == end_point: return
	
		end_point = 'Mixed_3b'
		self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_3c'
		self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'MaxPool3d_4a_3x3'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
			                                                 padding=0)
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_4b'
		self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_4c'
		self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_4d'
		self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_4e'
		self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_4f'
		self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'MaxPool3d_5a_2x2'
		self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
			                                                 padding=0) # was (2, 2, 2) for 224 x 224 input
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_5b'
		self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
		if self._final_endpoint == end_point: return

		end_point = 'Mixed_5c'
		self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
		if self._final_endpoint == end_point: return


		self.extracted_3c = Unit3D(
		    output_channels=480,
		    in_channels=480,
		    kernel_shape=[1, 1, 1],
		    stride=(4, 1, 1),
		    padding='SAME')

		self.extracted_4d = Unit3D(
		    output_channels=512,
		    in_channels=512,
		    kernel_shape=[1, 1, 1],
		    stride=(2, 1, 1),
		    padding='SAME')

		
		self.extracted_5c = Unit3D(
		    output_channels=1024,
		    in_channels=1024,
		    kernel_shape=[1, 1, 1],
		    stride=(1, 1, 1),
		    padding='SAME')
		

		self.concatenated_output_conv = Unit3D(
		    output_channels=1024,
		    in_channels=2016,
		    kernel_shape=[1, 1, 1],
		    stride=(1, 1, 1),
		    padding='SAME')

		self.maxPool3d_concatenated_2x2 = MaxPool3dSamePadding(
		    kernel_size=[2, 2, 2], stride=(2, 2, 2), padding='SAME')


		end_point = 'Logits'
		self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
			                         stride=(1, 1, 1))
		self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
			                 kernel_shape=[1, 1, 1],
			                 padding=0,
			                 activation_fn=None, #F.sigmoid, # was none changed to Sigmoid
			                 use_batch_norm=False,
			                 use_bias=True,
			                 name='logits')

		self.build()

	'''
	def replace_logits(self, num_classes):
		self._num_classes = num_classes
		self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
			                 kernel_shape=[1, 1, 1],
			                 padding=0,
			                 activation_fn=None,
			                 use_batch_norm=False,
			                 use_bias=True,
			                 name='logits')
	'''

	def replace_logits(self, num_classes):
		self._num_classes = num_classes
		self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
			                 kernel_shape=[1, 1, 1],
			                 padding=0,
			                 activation_fn=None,
			                 use_batch_norm=False,
			                 use_bias=True,
			                 name='logits')
		self.spatial_downsampling_layer = Unit3D(in_channels=1024, output_channels=1024, kernel_shape=[3, 3, 3], padding=1, stride = (1,2,2))
		self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 13],
			                         stride=(1, 1, 1))
	

	def build(self):
		for k in self.end_points.keys():
			self.add_module(k, self.end_points[k])

	def forward(self, x, tubes):
		input_shape = x.shape
		#print('input shape: ',input_shape)
		for end_point in self.VALID_ENDPOINTS:
			if end_point in self.end_points:
				x = self._modules[end_point](x) # use _modules to work with dataparallel
				if end_point == 'Mixed_3c':
					Mixed_3c = x
				if end_point == 'Mixed_4d':
					Mixed_4d = x
				if end_point== 'Mixed_5c':
					Mixed_5c = x
				#print(x.shape)

		#print('mixed 3c: ',Mixed_3c.size())
		#print('mixed 4d: ',Mixed_4d.size())
		#print('mixed 5c: ',Mixed_5c.size())
		#print('x: ',x.size())

		feature_mixed_3c = self.extract_feature( tubes, Mixed_3c, input_shape )
		feature_mixed_3c = torch.stack(feature_mixed_3c,dim=0)
		feature_mixed_3c = self.extracted_3c(feature_mixed_3c)

		feature_mixed_4d = self.extract_feature( tubes, Mixed_4d, input_shape )
		feature_mixed_4d = torch.stack(feature_mixed_4d,dim=0)
		feature_mixed_4d = self.extracted_4d(feature_mixed_4d)

		feature_mixed_5c = self.extract_feature( tubes, x, input_shape )
		feature_mixed_5c = torch.stack(feature_mixed_5c,dim=0)
		feature_mixed_5c = self.extracted_5c(feature_mixed_5c)

		concatenated_out = torch.cat((feature_mixed_3c, feature_mixed_4d, feature_mixed_5c),dim=1)
		out = self.concatenated_output_conv(concatenated_out)

		x = self.spatial_downsampling_layer(out)
		#print('after downsampling: ',x.size())

		x = self.logits(self.dropout(self.avg_pool(x)))
		#print("After logits:", x.shape)

		if self._spatial_squeeze:
			logits = x.squeeze(3).squeeze(3).squeeze(2)

		if self.multi_label:
			logits = torch.sigmoid(logits)

		# logits is batch X time X classes, which is what we want to work with
		return logits

	def reshape_tube(self,tube, height, width, expected_height, expected_width):
		new_tube = []
		##print('tube: ',tube)
		#print(height,'      ',width,'       ',expected_height,'     ',expected_width)
		ratio_h = expected_height/height
		ratio_w = expected_width/width
		new_tube = [tube[0] * ratio_w, tube[1]*ratio_h, tube[2] * ratio_w, tube[3]*ratio_h]

		return new_tube

	def randomly_generated_tube(self, max_height, max_width):                           
		t = [0, 0, random.randint(15,max_width-1), random.randint(8,max_height-1)]
		return t
		
	'''
	def replace_logits(self, num_classes):
		self.num_classes = num_classes
		self.output_classification_conv = Unit3Dpy(1024, self.num_classes, kernel_size=(3, 3, 3), activation=None, use_bias=True, use_bn=False)
	''' 

	def extract_feature(self, tube_maps, features, input_shape):   

		extracted_features = []
		#print('input_shape: ',input_shape)
		height = input_shape[3]
		width =  input_shape[4]
		#print(height,'  ',width,'   ',features.size()[0])
		
		for batch in range(features.size()[0]):
		    tube = []
		    tube_map = tube_maps[batch]
		    #tube_map = tube_map.data.cpu().numpy()
		    #print('tube map size: ',tube_map.size())
		    tube_points = (tube_map == 1).nonzero()
		    tube_points = tube_points.data.cpu().numpy()
		    #print('tube points: ',tube_points)

		    for point in tube_points:
		        tube.append(point[1])
		        tube.append(point[0])
		    
		    
		    #print('tube: ',tube)
		    feature = features[batch]
		    #print('feature size: ',feature.size())
		    expected_height = feature.size()[2]
		    expected_width = feature.size()[3]

		    if len(tube)!=0:
		        t = self.reshape_tube(tube, height, width, expected_height, expected_width)
		        #print('reshaped tube: ', t)
		    else:                                                                           #tube has nothing in case of videos with no actions. This case is handled here    
		        t = self.randomly_generated_tube(expected_height, expected_width)           # generating random tube 
		        #print('randomly generatd: ',t)
		    #print('reshaped tube: ',t)
		    t = [int(round(x)) for x in t]
		    #print('resized tube: ',t)
		    original_t = copy.deepcopy(t)
		    #########boundary conditions checked
		    if t[0]<0:
		        t[0]=0

		    if t[1]<0:
		        t[1]=0 

		    if t[2]>=expected_width:
		        t[2]=expected_width-1 

		    if t[3]>=expected_height:
		        t[3]=expected_height-1 

		    if t[0]>=expected_width:
		        t[0]=expected_width-1 

		    if t[1]>=expected_height:
		        t[1]=expected_height-1 
		    #####################################


		    extracted_part = feature[ :, :, t[1]:t[3]+1, t[0]:t[2]+1 ]
		    #print('extacted part: ',extracted_part.size())
		    

		    try:
		        extracted_part = F.interpolate(extracted_part, size=(14,25), mode='bilinear', align_corners=False)


		    except:
		        print('input and output sizes should be greater than 0')
		        print('extracted_output: ',extracted_part.size())
		        print('original t: ',original_t)
		        print('t: ',t)
		    #print('extacted part after interpolation: ',extracted_part.size())
		    extracted_features.append(extracted_part)

		return extracted_features

	def extract_features(self, x):
		for end_point in self.VALID_ENDPOINTS:
			if end_point in self.end_points:
			    x = self._modules[end_point](x)
		return self.avg_pool(x)


if __name__ == '__main__':
	model = InceptionI3d(num_classes=157, in_channels=3)
	model.load_state_dict(torch.load('/home/c3-0/praveen/VIRAT/trained_models/i3d_model_rgb_charades.pt'), strict= False)
	model.replace_logits(6)
	use_cuda=True
	if use_cuda:
		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
			model = nn.DataParallel(model)
			model.cuda()
			# criterion.cuda()

		else:
			print('only one gpu is being used')
			model.cuda()
	#model.cuda()


	print('loaded succesfully')

	tube = torch.zeros(1,448,800)
	tube[:,0,0] = 1
	tube[:,112,112] = 1
	tube.cuda()

	dummy_input = np.random.rand(1,3,16,448,800)
	dummy_input = np.array(dummy_input, dtype='f')
	dummy_input = torch.from_numpy(dummy_input).cuda()

	out = model(dummy_input, tube)

	print('out: ',out.size())
	exit()

	#exit()
	inputs = torch.randn(4, 3, 16, 448, 800).cuda()
	output = model(inputs)
	print(output.size())
