# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import pickle
import numpy as np
import random
import os
from utils import pad_seq, bytes_to_file, \
    read_split_image, shift_and_resize_image, normalize_image


class PickledImageProvider(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                    if len(examples) % 1000 == 0:
                        print("processed %d examples" % len(examples))
                except EOFError:
                    break
                except Exception:
                    pass
            print("unpickled total %d examples" % len(examples))
            return examples


def get_batch_iter(examples, batch_size):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            img_A = normalize_image(img_A)
            img_B = normalize_image(img_B)
            return np.concatenate([img_A[:,:,np.newaxis], img_B[:,:,np.newaxis]], axis=2)
        finally:
            img.close()

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            processed = [process(e[1]) for e in batch]
            # stack into tensor
            yield np.array(processed).astype(np.float32)

    return batch_iter()


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None):
        self.data_dir = data_dir
        self.filter_by = filter_by
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path)
        self.val = PickledImageProvider(self.val_path)
        if self.filter_by:
            print("filter by label ->", filter_by)
            self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
            self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)
        print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size)

    def get_train_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        while True:
            train_val_batch_iter = get_batch_iter(training_examples, batch_size)
            for examples in train_val_batch_iter:
                yield examples

    def get_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size)
            for examples in val_batch_iter:
                yield examples

    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    def get_train_val_path(self):
        return self.train_path, self.val_path


class InjectDataProvider(object):
    def __init__(self, obj_path):
        self.data = PickledImageProvider(obj_path)
        print("examples -> %d" % len(self.data.examples))

    def get_random_iter(self, batch_size):
        examples = self.data.examples[:]
        batch_iter = get_batch_iter(examples, batch_size)
        for images in batch_iter:
            yield images
