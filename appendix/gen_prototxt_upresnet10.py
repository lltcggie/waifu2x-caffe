import os
import os.path as osp
import sys
import google.protobuf as pb
from argparse import ArgumentParser
import numpy as np

import caffe
from caffe.proto import caffe_pb2


def _get_include(phase):
    inc = caffe_pb2.NetStateRule()
    if phase == 'train':
        inc.phase = caffe_pb2.TRAIN
    elif phase == 'test':
        inc.phase = caffe_pb2.TEST
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return inc


def _get_param(num_param):
    if num_param == 1:
        # only weight
        param = caffe_pb2.ParamSpec()
        param.lr_mult = 1
        param.decay_mult = 1
        return [param]
    elif num_param == 2:
        # weight and bias
        param_w = caffe_pb2.ParamSpec()
        param_w.lr_mult = 1
        param_w.decay_mult = 1
        param_b = caffe_pb2.ParamSpec()
        param_b.lr_mult = 2
        param_b.decay_mult = 0
        return [param_w, param_b]
    else:
        raise ValueError("Unknown num_param {}".format(num_param))


def Input(name, top, shape):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Input'
    layer.top.append(top)
    blobShape = layer.input_param.shape.add()
    blobShape.dim.extend(shape)
    return layer

def Conv(name, bottom, num_output, kernel_size, stride = 1, pad = 0, nobias = False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = 'msra'
    layer.convolution_param.bias_term = not(nobias)
    layer.param.extend(_get_param(1 if nobias else 2))
    return layer

def DeConv(name, bottom, num_output, kernel_size, stride = 1, pad = 0, nobias = False):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Deconvolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.weight_filler.type = 'msra'
    layer.convolution_param.bias_term = not(nobias)
    layer.param.extend(_get_param(1 if nobias else 2))
    return layer


def Sigmoid(name, bottom):
    top_name = name + '_sigmoid'
    # ReLU
    sigmoid_layer = caffe_pb2.LayerParameter()
    sigmoid_layer.name = name + '_sigmoid'
    sigmoid_layer.type = 'Sigmoid'
    sigmoid_layer.bottom.extend([bottom])
    sigmoid_layer.top.extend([top_name])
    return sigmoid_layer


def Relu(name, bottom):
    top_name = name + '_relu'
    # ReLU
    relu_layer = caffe_pb2.LayerParameter()
    relu_layer.name = name + '_relu'
    relu_layer.type = 'ReLU'
    relu_layer.bottom.extend([bottom])
    relu_layer.top.extend([top_name])
    return relu_layer


def LeakyRelu(name, bottom, negative_slope):
    top_name = name + '_relu'
    # LeakyRelu
    relu_layer = caffe_pb2.LayerParameter()
    relu_layer.name = name + '_relu'
    relu_layer.type = 'ReLU'
    relu_layer.relu_param.negative_slope = negative_slope
    relu_layer.bottom.extend([bottom])
    relu_layer.top.extend([top_name])
    return relu_layer


def ConvLeakyRelu(name, bottom, num_output, kernel_size, stride = 1, pad = 0, negative_slope = 0.1):
    layers = []
    layers.append(Conv(name, bottom, num_output, kernel_size, stride, pad))
    layers.append(LeakyRelu(name, layers[-1].top[0], negative_slope))
    return layers


def GlobalAvgPool(name, bottom, stride = 1, pad = 0):
    top_name = name + '_globalavgpool'
    layer = caffe_pb2.LayerParameter()
    layer.name = name + '_globalavgpool'
    layer.type = 'Pooling'
    layer.bottom.extend([bottom])
    layer.top.extend([top_name])
    layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    layer.pooling_param.stride = stride
    layer.pooling_param.pad = pad
    layer.pooling_param.global_pooling = True
    layer.pooling_param.engine = caffe_pb2.PoolingParameter.CAFFE
    return layer


def Linear(name, bottom, num_output):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'InnerProduct'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.inner_product_param.num_output = num_output
    layer.inner_product_param.weight_filler.type = 'msra'
    layer.inner_product_param.bias_filler.value = 0
    layer.param.extend(_get_param(2))
    return layer


def Crop(name, bottom, crop_size):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'CropCenter'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.crop_center_param.crop_size.extend(crop_size)
    return layer

def Add(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return layer

def Axpy(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Axpy'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return layer


def SEResBlock(name, bottom, in_channels, out_channels, r=16, slope=0.1):
    layers = []
    res_bottom = bottom

    layers.extend(ConvLeakyRelu(name + '/conv1', res_bottom, out_channels, 3, 1, 0, slope))
    layers.extend(ConvLeakyRelu(name + '/conv2', layers[-1].top[0], out_channels, 3, 1, 0, slope))
    h = layers[-1].top[0]

    layers.append(GlobalAvgPool(name + '/fc1', h))
    layers.append(Linear(name + '/fc1', layers[-1].top[0], out_channels // r))
    layers.append(Relu(name + '/fc1', layers[-1].top[0]))

    layers.append(Linear(name + '/fc2', layers[-1].top[0], out_channels))
    layers.append(Sigmoid(name + '/fc2', layers[-1].top[0]))
    se = layers[-1].top[0]

    layers.append(Crop(name + '/crop', res_bottom, [0, 0, 2, 2]))
    x = layers[-1].top[0]

    if in_channels != out_channels:
        layers.extend(ConvLeakyRelu(name + '/conv_bridge', x, out_channels, 3, 1, 0, slope))
        x = layers[-1].top[0]

    # h * se + x
    layers.append(Axpy(name + '/axpy', [se, h, x]))
    return layers


def create_model(ch):
    model = caffe_pb2.NetParameter()
    model.name = 'UpResNet10_{}'.format(ch)
    layers = []

    layers.append(Input('data', 'input', [1, ch, 90, 90]))

    layers.extend(ConvLeakyRelu('/conv_pre', layers[-1].top[0], 64, 3, 1, 0, 0.1))
    skip = layers[-1].top[0]

    layers.extend(SEResBlock('/res1', layers[-1].top[0], 64, 64, 4))
    layers.extend(SEResBlock('/res2', layers[-1].top[0], 64, 64, 4))
    layers.extend(SEResBlock('/res3', layers[-1].top[0], 64, 64, 4))
    layers.extend(SEResBlock('/res4', layers[-1].top[0], 64, 64, 4))
    layers.extend(SEResBlock('/res5', layers[-1].top[0], 64, 64, 4))

    layers.extend(ConvLeakyRelu('/conv_bridge', layers[-1].top[0], 64, 3, 1, 0, 0.1))
    h = layers[-1].top[0]

    layers.append(Crop('/crop', skip, [0, 0, 11, 11]))
    skip = layers[-1].top[0]

    layers.append(Add('/add', [h, skip]))

    layers.append(DeConv('/conv_post', layers[-1].top[0], ch, 4, 2, 3, True))

    model.layer.extend(layers)
    return model


def main(args):
    model = create_model(args.ch)
    if args.output is None:
        args.output = osp.join(osp.dirname(__file__),
            'upresnet10_{}.prototxt'.format(args.ch))
    with open(args.output, 'w') as f:
        f.write(pb.text_format.MessageToString(model))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ch', type=int, default=3,
                        choices=[1, 3])
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    main(args)
