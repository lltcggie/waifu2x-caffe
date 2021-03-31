import os
import os.path as osp
import sys
import google.protobuf as pb
from argparse import ArgumentParser
import numpy as np
import shutil

import caffe
from caffe.proto import caffe_pb2

sys.path.append('waifu2x-chainer')
from lib import srcnn
import chainer


def main():
    caffe.set_mode_cpu()

    model_name = 'UpResNet10'
    model_dir = 'waifu2x-chainer/models/{}'.format(model_name.lower())
    model_class = srcnn.archs[model_name]

    for filename in os.listdir(model_dir):
        basename, ext = os.path.splitext(filename)
        if ext == '.npz':
            model_path = os.path.join(model_dir, filename)
            print(model_path)
            channels = 3 if 'rgb' in filename else 1
            model = model_class(channels)
            chainer.serializers.load_npz(model_path, model)

            model.to_cpu()

            params = {}
            for path, param in model.namedparams():
                params[path] = param.array

            net = caffe.Net('upresnet10_3.prototxt', caffe.TEST)
            for key in net.params:
                l = len(net.params[key])
                net.params[key][0].data[...] = params[key + '/W']
                if l >= 2:
                    net.params[key][1].data[...] = params[key + '/b']

            input_data = np.empty(net.blobs['input'].data.shape, dtype=np.float32)
            input_data[...] = np.random.random_sample(net.blobs['input'].data.shape)

            net.blobs['input'].data[...] = input_data
            ret = net.forward()

            input_data = np.empty(net.blobs['input'].data.shape, dtype=np.float32)
            input_data[...] = np.random.random_sample(net.blobs['input'].data.shape)

            net.blobs['input'].data[...] = input_data
            ret = net.forward()

            batch_y = model(input_data)
            print(batch_y.array - ret['/conv_post'])

if __name__ == '__main__':
    caffe.init_log(3)
    main()
