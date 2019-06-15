# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import pickle
import numpy as np
import random
import os
from utils import pad_seq, bytes_to_file, \
    read_split_image, shift_and_resize_image, normalize_image

np.random.seed(3000)

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


def get_batch_iter(examples, batch_size, augment):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            img_A = normalize_image(img_A)
            img_B = normalize_image(img_B)
            if augment:
                r = random.randint(1,4)
                if r == 1:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                elif r == 2:
                    img_A = np.flipud(img_A)
                    img_B = np.flipud(img_B)
                elif r == 3:
                    img_A = np.rot90(img_A, k=2)
                    img_B = np.rot90(img_B, k=2)
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
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None, data_augmentation=False):
        self.data_dir = data_dir
        self.filter_by = filter_by
        self.data_augmentation = data_augmentation
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path)
        self.val = PickledImageProvider(self.val_path)
        self.val_spec = PickledImageProvider(self.val_path)
        if self.filter_by:
            print("filter by label ->", filter_by)
            self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
            self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)
        self.val_spec.examples = [e for e in self.val_spec.examples if e[0] == 1]
        print("train examples -> %d, val examples -> %d, special examples -> %d" % (len(self.train.examples), len(self.val.examples), len(self.val_spec.examples)))

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.seed(3000)
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size, augment=self.data_augmentation)

    def get_val_iter(self, batch_size, shuffle=True):
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.seed(3000)
            np.random.shuffle(val_examples)
        return get_batch_iter(val_examples, batch_size, augment=False)

    def get_val_spec_iter(self, batch_size, shuffle=True):
        val_spec_examples = self.val_spec.examples[:]
        if shuffle:
            np.random.seed(3000)
            np.random.shuffle(val_spec_examples)
        return get_batch_iter(val_spec_examples, batch_size, augment=False)

    def get_infinite_train_iter(self, batch_size, shuffle=True):
        """
        Training iterator runs forever
        """
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.seed(3000)
            np.random.shuffle(training_examples)
        while True:
            train_val_batch_iter = get_batch_iter(training_examples, batch_size, augment=self.data_augmentation)
            for examples in train_val_batch_iter:
                yield examples

    def get_infinite_val_iter(self, batch_size, shuffle=True):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.seed(3000)
            np.random.shuffle(val_examples)
        while True:
            val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
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
        batch_iter = get_batch_iter(examples, batch_size, augment=False)
        for images in batch_iter:
            yield images
