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

fname_convert_table = {
    'anime_style_noise0_scale_rgb': 'noise0_scale2.0x_model',
    'anime_style_noise1_scale_rgb': 'noise1_scale2.0x_model',
    'anime_style_noise2_scale_rgb': 'noise2_scale2.0x_model',
    'anime_style_noise3_scale_rgb': 'noise3_scale2.0x_model',
    'anime_style_scale_rgb': 'scale2.0x_model',
}

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
            size = 64 + model.offset
            data = np.zeros((1, channels, size, size), dtype=np.float32)
            x = chainer.Variable(data)
            chainer.serializers.load_npz(model_path, model)

            params = {}
            for path, param in model.namedparams():
                params[path] = param.array

            net = caffe.Net('upresnet10_3.prototxt', caffe.TEST)
            for key in net.params:
                l = len(net.params[key])
                net.params[key][0].data[...] = params[key + '/W']
                if l >= 2:
                    net.params[key][1].data[...] = params[key + '/b']
 
            prototxt_path = '{}.prototxt'.format(fname_convert_table[basename])
            caffemodel_path = '{}.json.caffemodel'.format(fname_convert_table[basename])
            net.save(caffemodel_path)
            shutil.copy('upresnet10_3.prototxt', prototxt_path)

if __name__ == '__main__':
    caffe.init_log(3)
    main()
