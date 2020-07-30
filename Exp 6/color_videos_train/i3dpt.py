import math
import os

import numpy as np
import torch
from torch.nn import ReplicationPad3d
import torch.nn as nn
import pdb
import torch.nn.functional as F



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


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True,
                 use_sigmoid=False):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        self.use_sigmoid = use_sigmoid
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

        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out


class BasicBlockBatchNorm(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1, type ='conv', output_padding=1, dilation=1, activation='relu', bias_initialization = 0, use_bn=True):
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
        if self.activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()


    def forward(self, x):
        out = self.convolution(x)
        if self.use_bn:
            out = self.bn(out)
        if self.activation_type is not None:
            out = self.activation(out)
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


class I3D(torch.nn.Module):
    def __init__(self,
                 num_classes=100,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D, self).__init__()

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

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2,2,2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

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





        # Beginning of deconvolution

        self.deconv4d = nn.ConvTranspose3d(1024, 256, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1)
        self.conv4f2 = Unit3Dpy(512, 256, kernel_size=(3, 3, 3), use_bn=False)


        self.deconv3c = nn.ConvTranspose3d(512, 256, kernel_size=(3,3,3), stride=(2,2,2), padding=1,  output_padding=(1,1, 1))
        self.conv3c2 = Unit3Dpy(480, 256, kernel_size=(3, 3, 3), use_bn=False)
        self.deconv3b = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=1, padding=1)


        self.deconv2c = nn.ConvTranspose3d(512, 96, kernel_size=(3,3,3), stride=(1,2,2), padding=1,  output_padding=(0,1, 1))
        self.conv1d = Unit3Dpy(96, 48, kernel_size=(3, 3, 3), use_bn=False)


        self.deconv1c = nn.ConvTranspose3d(48, 24, kernel_size=(3,3,3), stride=(1,2,2), padding=1,  output_padding=(0,1, 1))
        self.decoder_last_3 = BasicBlockBatchNorm(48, 8, kernel_size=3, stride=1, padding=3, dilation=3, use_bn=False)
        self.decoder_last_5 = BasicBlockBatchNorm(48, 8, kernel_size=3, stride=1, padding=5, dilation=5, use_bn=False)
        self.decoder_last_7 = BasicBlockBatchNorm(48, 8, kernel_size=3, stride=1, padding=7, dilation=7, use_bn=False)
        self.conv0d = Unit3Dpy(72, 1, kernel_size=(3, 3, 3), use_bn=False, activation=None)


        self.conv_fp1 = Unit3Dpy(256, 8, kernel_size=(3, 3, 3), use_bn=False)
        self.conv_fp2 = Unit3Dpy(256, 8, kernel_size=(3, 3, 3), use_bn=False)
        self.conv_fp3 = Unit3Dpy(96, 8, kernel_size=(3, 3, 3), use_bn=False)


        self.conv_MLLoss_1 = Unit3Dpy(256, 1, kernel_size=(3, 3, 3), use_bn=False, activation=None)
        self.conv_MLLoss_2 = Unit3Dpy(256, 1, kernel_size=(3, 3, 3), use_bn=False, activation=None)
        self.conv_MLLoss_3 = Unit3Dpy(96, 1, kernel_size=(3, 3, 3), use_bn=False, activation=None)




    def forward(self, inp):
        # Preprocessing
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        Mixed_3b = self.mixed_3b(out)
        Mixed_3c = self.mixed_3c(Mixed_3b)
        out = self.maxPool3d_4a_3x3(Mixed_3c)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        Mixed_4d = self.mixed_4d(out)
        out = self.mixed_4e(Mixed_4d)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        #pdb.set_trace()


        ## deconvolution stuff
        deconv4d = self.deconv4d(out) #
        feature_set_1 = self.conv_fp1(deconv4d)
        feature_set_1 = F.interpolate(feature_set_1, size=(8, 224, 400))
        MLLoss_output_1 = self.conv_MLLoss_1(deconv4d)


        conv4d2 = self.conv4f2(Mixed_4d)
        mixed_4h = torch.cat((conv4d2, deconv4d), 1)

        deconv3c = self.deconv3c(mixed_4h)
        feature_set_2 = self.conv_fp2(deconv3c)
        feature_set_2 = F.interpolate(feature_set_2, size=(8, 224, 400))
        MLLoss_output_2 = self.conv_MLLoss_2(deconv3c)


        conv3c2 = self.conv3c2(Mixed_3c)
        mixed_3d = torch.cat((conv3c2, deconv3c), 1)

        deconv3b = self.deconv3b(mixed_3d)
        mixed_3e = torch.cat((Mixed_3b, deconv3b), 1)


        deconv2c = self.deconv2c(mixed_3e)
        feature_set_3 = self.conv_fp3(deconv2c)
        feature_set_3 = F.interpolate(feature_set_3, size=(8, 224, 400))
        MLLoss_output_3 = self.conv_MLLoss_3(deconv2c)



        conv1d = self.conv1d(deconv2c)

        deconv1c = self.deconv1c(conv1d)
        penultimate_features = torch.cat((deconv1c, feature_set_1, feature_set_2, feature_set_3), 1)
        out_decoder_3 = self.decoder_last_3(penultimate_features)
        out_decoder_5 = self.decoder_last_5(penultimate_features)
        out_decoder_7 = self.decoder_last_7(penultimate_features)
        out_last = torch.cat((penultimate_features, out_decoder_3, out_decoder_5, out_decoder_7), 1)
        conv0d = self.conv0d(out_last)
        return conv0d, MLLoss_output_1, MLLoss_output_2, MLLoss_output_3, Mixed_3c



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
        print("TEST STEP",i)

        dummy_input = np.random.rand(2,3,16,448,800)
        dummy_input = np.array(dummy_input, dtype='f')
        dummy_input = torch.from_numpy(dummy_input).cuda()

        dummy_target = np.random.rand(2,1,8,224,400)
        dummy_target = np.array(dummy_target, dtype='f')
        dummy_target = torch.from_numpy(dummy_target).cuda()


        #pdb.set_trace()
        criterion = nn.BCELoss(size_average=True).cuda()

        model = I3D()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        model.train()
        model = model.cuda()

        output, MLLoss_output_1, MLLoss_output_2, MLLoss_output_3 = model(dummy_input)
        print(output.shape)

        loss = criterion(torch.sigmoid(output), dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
