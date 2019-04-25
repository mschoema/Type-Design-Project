import tensorflow as tf
import numpy as np
from ops import *
import pickle as pickle
import scipy.misc as misc
import random
import os
from utils import *
from collections import namedtuple
import time
from array_display import display_array

GENERATOR_DIM = 64
WIDTH = 256
HEIGHT = 256
SIZE = (WIDTH, HEIGHT)
BATCH_SIZE = 2
INPUT_FILTERS = 1
OUTPUT_FILTERS = 1
LOSS_DEPTH = 1

LossHandle = namedtuple("LossHandle", ["edge_loss"])
InputHandle = namedtuple("InputHandle", ["real_data"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source", "edges", "all_edges"])
SummaryHandle = namedtuple("SummaryHandle", ["g_merged"])

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
            img_A, img_B, loss_map = read_split_image(img)
            img_A = normalize_image(img_A)
            img_B = normalize_image(img_B)
            loss_map = loss_map*img_A
            return np.concatenate([img_A[:,:,np.newaxis], img_B[:,:,np.newaxis], loss_map[:,:,np.newaxis]], axis=2)
        finally:
            img.close()

    def batch_iter():
        for i in range(0, len(padded), batch_size):
            for j in range(1):
                batch = padded[i: i + batch_size]
                processed = [process(e[1]) for e in batch]
                # stack into tensor
                yield np.array(processed).astype(np.float32)

    return batch_iter()

class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.obj", val_name="val.obj"):
        self.data_dir = data_dir
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)
        self.train = PickledImageProvider(self.train_path)
        self.val = PickledImageProvider(self.val_path)

    def get_train_iter(self, batch_size, shuffle=True):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
        return get_batch_iter(training_examples, batch_size)

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

def encoder(images, is_training, reuse=False):
    with tf.variable_scope("generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        encode_layers = dict()

        def encode_layer(x, output_filters, layer):
            act = lrelu(x)
            conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
            enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
            encode_layers["e%d" % layer] = enc
            return enc

        e1 = conv2d(images, GENERATOR_DIM, scope="g_e1_conv")
        encode_layers["e1"] = e1
        e2 = encode_layer(e1, GENERATOR_DIM * 2, 2)
        e3 = encode_layer(e2, GENERATOR_DIM * 4, 3)
        e4 = encode_layer(e3, GENERATOR_DIM * 8, 4)
        e5 = encode_layer(e4, GENERATOR_DIM * 8, 5)
        e6 = encode_layer(e5, GENERATOR_DIM * 8, 6)
        e7 = encode_layer(e6, GENERATOR_DIM * 8, 7)
        e8 = encode_layer(e7, GENERATOR_DIM * 8, 8)

        return e8, encode_layers

def decoder(encoded, encoding_layers, is_training, reuse=False):
    with tf.variable_scope("generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s = WIDTH
        s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
            s / 64), int(s / 128)

        def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
            dec = deconv2d(tf.nn.relu(x), [BATCH_SIZE, output_width,
                                           output_width, output_filters], scope="g_d%d_deconv" % layer)
            if layer != 8:
                dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
            if dropout:
                dec = tf.nn.dropout(dec, 0.5)
            if do_concat:
                dec = tf.concat([dec, enc_layer], 3)
            return dec

        d1 = decode_layer(encoded, s128, GENERATOR_DIM * 8, layer=1, enc_layer=encoding_layers["e7"],
                          dropout=True)
        d2 = decode_layer(d1, s64, GENERATOR_DIM * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
        d3 = decode_layer(d2, s32, GENERATOR_DIM * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
        d4 = decode_layer(d3, s16, GENERATOR_DIM * 8, layer=4, enc_layer=encoding_layers["e4"])
        d5 = decode_layer(d4, s8, GENERATOR_DIM * 4, layer=5, enc_layer=encoding_layers["e3"])
        d6 = decode_layer(d5, s4, GENERATOR_DIM * 2, layer=6, enc_layer=encoding_layers["e2"])
        d7 = decode_layer(d6, s2, GENERATOR_DIM, layer=7, enc_layer=encoding_layers["e1"])
        d8 = decode_layer(d7, s, OUTPUT_FILTERS, layer=8, enc_layer=None, do_concat=False)

        output = tf.nn.tanh(d8)  # scale to (-1, 1)
        return output

def edge_computation(decoded):
    sigm = (-1.*decoded + 1.)/2.
    xor, all_edges = xor_pool2d3x3(sigm)
    return xor, sigm

def retrieve_generator_vars():
    all_vars = tf.global_variables()
    generate_vars = [var for var in all_vars if "g_" in var.name]
    return generate_vars

def generator(images, is_training, reuse=False):
    e8, enc_layers = encoder(images, is_training=is_training, reuse=reuse)
    output = decoder(e8, enc_layers, is_training=is_training, reuse=reuse)
    edges, all_edges = edge_computation(output)
    return output, e8, edges, all_edges

def build_model(is_training=True):
        real_data = tf.placeholder(tf.float32,
                                   [BATCH_SIZE, WIDTH, WIDTH,
                                    INPUT_FILTERS + OUTPUT_FILTERS + LOSS_DEPTH],
                                   name='real_A_and_B_images')

        # target images
        real_B = real_data[:, :, :, :INPUT_FILTERS]
        # source images
        real_A = real_data[:, :, :, INPUT_FILTERS:INPUT_FILTERS + OUTPUT_FILTERS]

        loss_maps = real_data[:, :, :, INPUT_FILTERS + OUTPUT_FILTERS:INPUT_FILTERS + OUTPUT_FILTERS + LOSS_DEPTH]

        fake_B, encoded_real_A, edges_fake_B, all_edges = generator(real_A, is_training=is_training)

        # Edge loss of generated images
        edge_loss = dist_map_loss(loss_maps, edges_fake_B)*1 + tf.reduce_mean(tf.abs(fake_B - real_B))*0

        edge_loss_summary = tf.summary.scalar("edge_loss", edge_loss)
        g_merged_summary = tf.summary.merge([edge_loss_summary])

        input_handle = InputHandle(real_data=real_data)

        loss_handle = LossHandle(edge_loss=edge_loss)

        eval_handle = EvalHandle(encoder=encoded_real_A,
                                 generator=fake_B,
                                 target=real_B,
                                 source=real_A,
                                 edges=edges_fake_B,
                                 all_edges=all_edges)

        summary_handle = SummaryHandle(g_merged=g_merged_summary)

        return input_handle, loss_handle, eval_handle, summary_handle

def generate_fake_samples(sess, input_handle, loss_handle, eval_handle, input_images):
        fake_images, real_images, edge_images, all_edges, \
        edge_loss = sess.run([eval_handle.generator,
                                 eval_handle.target,
                                 eval_handle.edges,
                                 eval_handle.all_edges,
                                 loss_handle.edge_loss],
                                feed_dict={input_handle.real_data: input_images})
        return fake_images, real_images, edge_images, all_edges, edge_loss

def validate_model(sess, sample_dir, val_iter, input_handle, loss_handle, eval_handle, epoch, step):
        images = next(val_iter)
        fake_imgs, real_imgs, edge_images, all_edges, edge_loss = generate_fake_samples(sess, input_handle, loss_handle, eval_handle, images)
        print("Sample: edge_loss: %.5f" % (edge_loss))

        # merged_all_edge_images = merge(np.round(all_edges), [BATCH_SIZE, 1])
        # merged_edge_images = merge(np.round(edge_images), [BATCH_SIZE, 1])
        merged_fake_images = merge(np.round(scale_back(fake_imgs)), [BATCH_SIZE, 1])
        merged_real_images = merge(scale_back(real_imgs), [BATCH_SIZE, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

        model_sample_dir = os.path.join(sample_dir, "model1_0")
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.png" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair)

def train(sess, input_handle, loss_handle, eval_handle, summary_handle, lr=0.0002, epoch=100, schedule=10, sample_steps=50):
        g_vars = retrieve_generator_vars()

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.edge_loss, var_list=g_vars)
        tf.global_variables_initializer().run()
        real_data = input_handle.real_data

        # filter by one type of labels
        data_provider = TrainDataProvider("../inputFiles/experiment/")
        total_batches = data_provider.compute_total_batch_num(BATCH_SIZE)
        val_batch_iter = data_provider.get_val_iter(BATCH_SIZE)

        experiment_dir = "../outputFiles/"
        checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
        log_dir = os.path.join(experiment_dir, "logs")
        if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print("create log directory")
        sample_dir = os.path.join(experiment_dir, "sample")
        if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
                print("create sample directory")

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(BATCH_SIZE)

            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                im = batch[0]
                counter += 1
                # Optimize G
                _, batch_edge_loss, g_summary = sess.run([g_optimizer, loss_handle.edge_loss, summary_handle.g_merged], feed_dict={real_data: batch,learning_rate: current_lr})
                
                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, edge_loss: %.5f"
                print(log_format % (ei, bid+1, total_batches, passed, batch_edge_loss))
                summary_writer.add_summary(g_summary, counter)

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    validate_model(sess, sample_dir, val_batch_iter, input_handle, loss_handle, eval_handle, ei, counter)

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        input_handle, loss_handle, eval_handle, summary_handle = build_model()
        train(sess, input_handle, loss_handle, eval_handle, summary_handle, lr=0.0002, epoch=50, schedule=10, sample_steps=5)

if __name__ == "__main__":
    main()