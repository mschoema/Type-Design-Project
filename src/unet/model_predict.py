# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import argparse

from unet import UNet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='Prediction for unseen data')
parser.add_argument('--model_dir', dest='model_dir', required=True,
                    help='directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--source_obj', dest='source_obj', type=str, required=True, help='the source images for inference')
parser.add_argument('--save_dir', default='save_dir', type=str, required=True, help='path to save inferred images')
args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size)
        model.register_session(sess)
        model.build_model(is_training=False)
        model.infer(model_dir=args.model_dir, source_obj=args.source_obj, save_dir=args.save_dir)

if __name__ == '__main__':
    tf.app.run()
