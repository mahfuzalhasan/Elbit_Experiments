import math
import os

import numpy as np
import torch
from torch.nn import ReplicationPad3d
import torch.nn as nn
import pdb
import torch.nn.functional as F
import pdb
import random
import copy


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class BatchRenormalization3D(torch.nn.Module):
    def __init__(self, num_features,  eps=1e-05, momentum=0.01, r_d_max_inc_step = 0.0001):
        super(BatchRenormalization3D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor((momentum), requires_grad = False)

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1, 1)), requires_grad=True)

        self.running_avg_mean = torch.nn.Parameter(torch.ones((1, num_features, 1, 1, 1)), requires_grad=False)
        self.running_avg_std = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1, 1)), requires_grad=False)

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = torch.tensor((1.0), requires_grad = False)
        self.d_max = torch.tensor((0.0), requires_grad = False)

    def forward(self, x):

        device = x.device

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3, 4), keepdim=True).to(device)
        batch_ch_std = torch.clamp(torch.std(x, dim=(0, 2, 3, 4), keepdim=True), self.eps, 1e10).to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        self.momentum = self.momentum.to(device)
        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)

        if self.training:
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).to(device).data.to(device)
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).to(device).data.to(device)

            x = ((x - batch_ch_mean) * r )/ batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]

        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        self.running_avg_mean.add_(self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean))
        self.running_avg_std.add_(self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std))
        
        return x


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='elu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)
            # self.batch3d = BatchRenormalization3D(out_channels)

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation == 'relu':
            out = torch.nn.functional.relu(out)
        elif self.activation == 'leaky_relu':
            out = torch.nn.functional.leaky_relu_(out, negative_slope=0.1)
        elif self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation == 'elu':
            out = torch.nn.functional.elu(out)
        return out



class BasicBlockBatchNorm(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1, type ='conv', output_padding=1, dilation=1, activation='elu', bias_initialization = 0, use_bn=True):
        super(BasicBlockBatchNorm, self).__init__()
        self.activation_type = activation
        self.use_bn = use_bn
        if type == 'conv':
            self.convolution = nn.Conv3d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        elif type == 'upsample':
            self.convolution = nn.ConvTranspose3d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, bias=False)
        nn.init.xavier_normal_(self.convolution.weight)
        #nn.init.constant_(self.convolution.bias, bias_initialization)
        if self.use_bn:
            self.bn = nn.BatchNorm3d(outplanes)
            # self.bn = BatchRenormalization3D(outplanes)

    def forward(self, x):
        out = self.convolution(x)
        if self.use_bn:
            out = self.bn(out)
        if self.activation_type == 'relu':
            out = torch.nn.functional.relu(out)
        elif self.activation_type == 'leaky_relu':
            out = torch.nn.functional.leaky_relu_(out, negative_slope=0.1)
        elif self.activation_type == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation_type == 'elu':
            out = torch.nn.functional.elu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out



class I3D_partial(torch.nn.Module):
	def __init__(self,
		         num_classes=36,
		         modality='rgb',
		         dropout_prob=0,
		         name='inception'):
		super(I3D_partial, self).__init__()

		self.name = name
		self.num_classes = num_classes
		if modality == 'rgb':
		    in_channels = 3
		elif modality == 'flow':
		    in_channels = 2
		else:
		    raise ValueError(
		        '{} not among known modalities [rgb|flow]'.format(modality))
		self.modality = modality

		self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')


		# Mixed 4
		self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
		self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
		self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
		self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
		self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])


		self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
		    kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')


		# Mixed 5
		self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
		self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])


		self.encoder_last_3 = BasicBlockBatchNorm(1024, 128, kernel_size=3, stride=1, padding=2, dilation=2)
		self.encoder_last_5 = BasicBlockBatchNorm(1024, 128, kernel_size=3, stride=1, padding=3, dilation=3)
		self.encoder_last_7 = BasicBlockBatchNorm(1024, 128, kernel_size=3, stride=1, padding=4, dilation=4)

		self.encoder_last_1x1 = BasicBlockBatchNorm(1408, 1024, kernel_size=3, stride=1, padding=1)

		self.spatial_downsampling_layer = Unit3Dpy(1024, 1024, kernel_size=(3, 3, 3), stride = (1,2,2))
		self.output_classification_conv = Unit3Dpy(1024, self.num_classes, kernel_size=(3, 3, 3), activation=None, use_bias=True, use_bn=False)
		self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 12], stride=(1, 1, 1))


		self.dropout = nn.Dropout(0.5)
		self.sigmoid = nn.Sigmoid()

	def reshape_tube(self,tube, height, width, expected_height, expected_width):
		new_tube = []
		###print('tube: ',tube)
		##print(height,'      ',width,'       ',expected_height,'     ',expected_width)
		ratio_h = expected_height/height
		ratio_w = expected_width/width
		new_tube = [tube[0] * ratio_w, tube[1]*ratio_h, tube[2] * ratio_w, tube[3]*ratio_h]

		return new_tube

	def randomly_generated_tube(self, max_height, max_width):                           
		t = [0, 0, random.randint(15,max_width-1), random.randint(8,max_height-1)]        #python random.randint(0,n)---> generate any number in [0,n]. It's not numpy random 
		                                                                              #15 and 8 are handpicked number. Lowest feature selection are from last layer which is 
		                                                                              # of (b,1024,2,14,25) size. So, idea is to choose starting point of random number in a way so that
		                                                                              # boxes have subsrtantial amount in both height and width dimension 

		return t
		
		

	def extract_feature(self, tube_maps, features, input_shape):   

		extracted_features = []
		##print('input_shape: ',input_shape)
		height = input_shape[3]
		width =  input_shape[4]
		##print(height,'  ',width,'   ',features.size()[0])
		
		for batch in range(features.size()[0]):
		    tube = []
		    tube_map = tube_maps[batch]
		    #tube_map = tube_map.data.cpu().numpy()
		    ##print('tube map size: ',tube_map.size())
		    tube_points = (tube_map == 1).nonzero()
		    tube_points = tube_points.data.cpu().numpy()
		    ##print('tube points: ',tube_points)

		    for point in tube_points:
		        tube.append(point[1])
		        tube.append(point[0])
		    
		    
		    ##print('tube: ',tube)
		    feature = features[batch]
		    #print('feature size: ',feature.size())
		    expected_height = feature.size()[2]
		    expected_width = feature.size()[3]

		    if len(tube)!=0:
		        t = self.reshape_tube(tube, 448, 800, expected_height, expected_width)
		        ##print('reshaped tube: ', t)
		    else:                                                                           #tube has nothing in case of videos with no actions. This case is handled here    
		        t = self.randomly_generated_tube(expected_height, expected_width)           # generating random tube 
		        ##print('randomly generatd: ',t)
		    ##print('reshaped tube: ',t)
		    t = [int(round(x)) for x in t]
		    ##print('resized tube: ',t)
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
		    ##print('extacted part: ',extracted_part.size())
		    

		    try:
		        extracted_part = F.interpolate(extracted_part, size=(56,100), mode='bilinear', align_corners=False)


		    except:
		        print('input and output sizes should be greater than 0')
		        print('extracted_output: ',extracted_part.size())
		        print('original t: ',original_t)
		        print('t: ',t)
		    
		    extracted_features.append(extracted_part)

		return extracted_features


	def replace_logits(self, num_classes):
		self.num_classes = num_classes
		self.output_classification_conv = Unit3Dpy(1024, self.num_classes, kernel_size=(3, 3, 3), activation=None, use_bias=True, use_bn=False)      

	def forward(self, inp, tubes):
		# Preprocessing
		
		input_shape = inp.shape
	   
		Mixed_3c = self.extract_feature(tubes,inp,input_shape)  #discard batch loop
		Mixed_3c = torch.stack(Mixed_3c,dim=0)  #   
		##print('mixed 3c: ',Mixed_3c.shape)
		out = self.maxPool3d_4a_3x3(Mixed_3c) # spatial 16, temp 4

		##print('out before inception mixed_4b: ',out.size())
		out = self.mixed_4b(out)
		##print('out from inception mixed_4b: ',out.size())
		out = self.mixed_4c(out)
		##print('out from inception mixed_4c: ',out.size())
		Mixed_4d = self.mixed_4d(out)
		##print('out from inception mixed_4d: ',Mixed_4d.size())
		out = self.mixed_4e(Mixed_4d)
		##print('out from inception mixed_4e: ',out.size())
		out = self.mixed_4f(out)
		##print('out from inception mixed_4f: ',out.size())
		out = self.maxPool3d_5a_2x2(out)
		##print('out from maxpool_5a: ',out.size())
		out = self.mixed_5b(out)
		##print('out from inception mixed_5b: ',out.size())
		out = self.mixed_5c(out)  ## [1, 1024, 2, 14, 25]
		##print('out from inception mixed_5c: ',out.size())

		out_encoder_3 = self.encoder_last_3(out)
		out_encoder_5 = self.encoder_last_5(out)
		out_encoder_7 = self.encoder_last_7(out)
		out = torch.cat(( out, out_encoder_3, out_encoder_5, out_encoder_7 ), 1 )
		out = self.encoder_last_1x1(out)
		##print('out size: ',out.size())
		

		###################classification
		output_classification = self.spatial_downsampling_layer(out)
		output_classification = self.dropout(self.avg_pool(output_classification))
		logits  = self.output_classification_conv(output_classification)
		logits = logits.squeeze(3).squeeze(3).squeeze(2)
		logits = self.sigmoid(logits)
		
		#################################

		
		return logits



	def load_tf_weights(self, sess):
		state_dict = {}
		if self.modality == 'rgb':
		    prefix = 'RGB/inception_i3d'
		elif self.modality == 'flow':
		    prefix = 'Flow/inception_i3d'
		load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
		            os.path.join(prefix, 'Conv3d_1a_7x7'))
		load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
		            os.path.join(prefix, 'Conv3d_2b_1x1'))
		load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
		            os.path.join(prefix, 'Conv3d_2c_3x3'))

		load_mixed(state_dict, 'mixed_3b', sess,
		           os.path.join(prefix, 'Mixed_3b'))
		load_mixed(state_dict, 'mixed_3c', sess,
		           os.path.join(prefix, 'Mixed_3c'))
		load_mixed(state_dict, 'mixed_4b', sess,
		           os.path.join(prefix, 'Mixed_4b'))
		load_mixed(state_dict, 'mixed_4c', sess,
		           os.path.join(prefix, 'Mixed_4c'))
		load_mixed(state_dict, 'mixed_4d', sess,
		           os.path.join(prefix, 'Mixed_4d'))
		load_mixed(state_dict, 'mixed_4e', sess,
		           os.path.join(prefix, 'Mixed_4e'))
		# Here goest to 0.1 max error with tf
		load_mixed(state_dict, 'mixed_4f', sess,
		           os.path.join(prefix, 'Mixed_4f'))

		load_mixed(
		    state_dict,
		    'mixed_5b',
		    sess,
		    os.path.join(prefix, 'Mixed_5b'),
		    fix_typo=True)
		load_mixed(state_dict, 'mixed_5c', sess,
		           os.path.join(prefix, 'Mixed_5c'))
		load_conv3d(
		    state_dict,
		    'conv3d_0c_1x1',
		    sess,
		    os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
		    bias=True,
		    bn=False)
		self.load_state_dict(state_dict)


def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt +
                   '.batch3d.bias'] = torch.from_numpy(beta.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var.squeeze())


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
                os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))



if __name__ == "__main__":


	import numpy as np


	for i in range(10):


		##print("TEST STEP",i)

		'''
		#tube = [[298.33333333 , 149.33333333,  370.83333333, 236.85925926],[529.58333333, 111.58518519, 568.33333333, 164.68148148]]
		tube = [[298.33333333 , 149.33333333,  370.83333333, 236.85925926],[]]
		tube = np.asarray(tube, dtype='f')
		tube = torch.from_numpy(tube).cuda()
		'''
		#tube = [torch.tensor([298.3333, 149.3333, 370.8333, 236.8593]), torch.tensor([])]
		tube = torch.zeros(1,448,800)
		tube[:,0,0] = 1
		tube[:,56,56] = 1
		tube.cuda()
		dummy_input = np.random.rand(1,480,8,56,100)
		dummy_input = np.array(dummy_input, dtype='f')
		dummy_input = torch.from_numpy(dummy_input).cuda()

		dummy_target = np.random.rand(1,1,8,224,400)
		dummy_target = np.array(dummy_target, dtype='f')
		dummy_target = torch.from_numpy(dummy_target).cuda()


		#pdb.set_trace()
		criterion = nn.BCELoss(size_average=True).cuda()

		model = I3D_partial()
		saved_model_file = "/home/c3-0/mahfuz/MEVA_results/models/04-07-20_2121/model_27.pth"
		model.load_state_dict(torch.load(saved_model_file)['state_dict_cls'])
		model.replace_logits(6)
		use_cuda = True
		if use_cuda:
			if torch.cuda.device_count() > 1:
				print("Let's use", torch.cuda.device_count(), "GPUs!")
				# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
				model = nn.DataParallel(model)
				#model.cuda()

				#model = nn.DataParallel(model)
				#model_classification.cuda()
				# criterion.cuda()

			else:
				print('only one gpu is being used')
				#model_classification.cuda()

		#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
		#saved_model_file = "/home/c3-0/mahfuz/MEVA_results/models/04-07-20_2121/model_27.pth"
		#model.load_state_dict(torch.load(saved_model_file)['state_dict_cls'])
		#model.replace_logits(6)
		#model.train()
		model = model.cuda()

		logits = model(dummy_input, tube)
		###print('outputs size: ',conv0d.size())
		###print('output1 size: ',MLLoss_output_1.size())
		###print('output2 size: ',MLLoss_output_2.size())
		###print('output3 size: ',MLLoss_output_3.size())
		print('logits size: ', logits.size())
		print('logits: ', logits)
		
		exit()
		loss = criterion(torch.sigmoid(output), dummy_target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
