# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import pickle
import os
import time
from collections import namedtuple
from ops import conv2d, deconv2d, lrelu, fc, batch_norm
from dataset import TrainDataProvider, InjectDataProvider
from utils import scale_back, merge, save_concat_images

# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["g_loss", "const_loss", "l1_loss", "l2_edge_loss"])
InputHandle = namedtuple("InputHandle", ["real_data"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source"])
SummaryHandle = namedtuple("SummaryHandle", ["g_merged"])


class UNet(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 generator_dim=64, L1_penalty=100, L2_edge_penalty=15, Lconst_penalty=15, dropout=False, input_filters=1, output_filters=1, data="data"):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.generator_dim = generator_dim
        self.L1_penalty = L1_penalty
        self.L2_edge_penalty = L2_edge_penalty
        self.Lconst_penalty = Lconst_penalty
        self.dropout = dropout
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.data = data
        self.counter_list = []
        self.train_l1_loss_list = []
        self.train_iou_list = []
        self.val_l1_loss_list = []
        self.val_iou_list = []
        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, self.data)
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")
            self.lists_dir = os.path.join(self.experiment_dir, "lists")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")
            if not os.path.exists(self.lists_dir):
                os.makedirs(self.lists_dir)
                print("create lists directory")

    def encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer, dropout=False):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                if dropout:
                    enc = tf.nn.dropout(enc, 0.5)
                encode_layers["e%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="g_e1_conv")
            encode_layers["e1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2, dropout=self.dropout)
            e3 = encode_layer(e2, self.generator_dim * 4, 3, dropout=self.dropout)
            e4 = encode_layer(e3, self.generator_dim * 8, 4, dropout=self.dropout)
            e5 = encode_layer(e4, self.generator_dim * 8, 5, dropout=self.dropout)
            e6 = encode_layer(e5, self.generator_dim * 8, 6, dropout=self.dropout)
            e7 = encode_layer(e6, self.generator_dim * 8, 7, dropout=self.dropout)
            e8 = encode_layer(e7, self.generator_dim * 8, 8, dropout=self.dropout)

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width,
                                               output_width, output_filters], scope="g_d%d_deconv" % layer)
                if layer != 8:
                    dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec

            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e7"],
                              dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e4"], dropout=self.dropout)
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e3"], dropout=self.dropout)
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e2"], dropout=self.dropout)
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["e1"], dropout=self.dropout)
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            return output

    def generator(self, images, is_training, reuse=False):
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)
        output = self.decoder(e8, enc_layers, is_training=is_training, reuse=reuse)
        return output, e8

    def edgeDetectionLayer(self, images):
        (batch_size, h, w, d) = images.shape
        edges = tf.image.sobel_edges(images)
        edges = tf.reshape(edges, (batch_size, h, w, 2*d))
        return edges

    def build_model(self, is_training=True, no_target_source=False):
        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_width, self.input_width,
                                    self.input_filters + self.output_filters],
                                   name='real_A_and_B_images')

        # target images
        real_B = real_data[:, :, :, :self.input_filters]
        # source images
        real_A = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]

        fake_B, encoded_real_A = self.generator(real_A, is_training=is_training)
        real_AB = tf.concat([real_A, real_B], 3)
        fake_AB = tf.concat([real_A, fake_B], 3)

        edges_fake_B = self.edgeDetectionLayer(fake_B)
        edges_real_B = self.edgeDetectionLayer(real_B)

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
        const_loss = (tf.reduce_mean(tf.square(encoded_real_A - encoded_fake_B))) * self.Lconst_penalty

        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))

         # L2 loss between eges of real and generated images
        l2_edge_loss = self.L2_edge_penalty * tf.reduce_mean(tf.square(edges_fake_B - edges_real_B))

        g_loss = l1_loss + l2_edge_loss + const_loss

        l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
        l2_edge_loss_summary = tf.summary.scalar("l2_edge_loss", l2_edge_loss)
        const_loss_summary = tf.summary.scalar("const_loss", const_loss)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)

        g_merged_summary = tf.summary.merge([l1_loss_summary, l2_edge_loss_summary,
                                            const_loss_summary, g_loss_summary])

        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data)

        loss_handle = LossHandle(g_loss=g_loss,
                                 const_loss=const_loss,
                                 l1_loss=l1_loss,
                                 l2_edge_loss=l2_edge_loss)

        eval_handle = EvalHandle(encoder=encoded_real_A,
                                 generator=fake_B,
                                 target=real_B,
                                 source=real_A)

        summary_handle = SummaryHandle(g_merged=g_merged_summary)

        # those operations will be shared, so we need
        # to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        t_vars = tf.trainable_variables()

        g_vars = [var for var in t_vars if 'g_' in var.name]

        if freeze_encoder:
            # exclude encoder weights
            print("freeze encoder weights")
            g_vars = [var for var in g_vars if not ("g_e" in var.name)]

        return g_vars

    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generate_vars = [var for var in all_vars if "g_" in var.name]
        return generate_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")

        return input_handle, loss_handle, eval_handle, summary_handle

    def get_model_id_and_dir(self):
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        model_name = "unet.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images):
        input_handle, loss_handle, eval_handle, summary_handle = self.retrieve_handles()
        fake_images, real_images, source_images, \
        g_loss, l1_loss, l2_edge_loss = self.sess.run([eval_handle.generator,
                                                 eval_handle.target,
                                                 eval_handle.source,
                                                 loss_handle.g_loss,
                                                 loss_handle.l1_loss,
                                                 loss_handle.l2_edge_loss],
                                                feed_dict={
                                                    input_handle.real_data: input_images
                                                })
        return fake_images, real_images, source_images, g_loss, l1_loss, l2_edge_loss

    def sample_model(self, val_iter, epoch, step, is_train_data=False, is_special_data=False):
        images = next(val_iter)
        fake_imgs, real_imgs, source_images, g_loss, l1_loss, l2_edge_loss = self.generate_fake_samples(images)
        if is_train_data:
            print("Train sample: g_loss: %.5f, l1_loss: %.5f, l2_edge_loss %.5f" % (g_loss, l1_loss, l2_edge_loss))
        elif is_special_data:
            print("Special sample: g_loss: %.5f, l1_loss: %.5f, l2_edge_loss %.5f" % (g_loss, l1_loss, l2_edge_loss))
        else:
            print("Val sample: g_loss: %.5f, l1_loss: %.5f, l2_edge_loss %.5f" % (g_loss, l1_loss, l2_edge_loss))

        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_source_images = merge(scale_back(source_images), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images, merged_source_images], axis=1)

        model_id, _ = self.get_model_id_and_dir()

        model_sample_dir = os.path.join(self.sample_dir, model_id)
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        if is_train_data:
            sample_img_path = os.path.join(model_sample_dir, "train_sample_%02d_%04d.png" % (epoch, step))
        elif is_special_data:
            sample_img_path = os.path.join(model_sample_dir, "special_sample_%02d.png" % (epoch))
        else:
            sample_img_path = os.path.join(model_sample_dir, "val_sample_%02d_%04d.png" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair[:,:,0])

    def validate_model(self, data_provider):
        train_iter = data_provider.get_train_iter(self.batch_size)
        val_iter = data_provider.get_val_iter(self.batch_size)

        total_train_l1_loss = 0
        total_train_iou = 0
        total_val_l1_loss = 0
        total_val_iou = 0

        train_count = 0
        for bid, batch in enumerate(train_iter):
            train_count += 1
            fake_imgs, real_imgs, _, _, l1_loss, _ = self.generate_fake_samples(batch)
            binary_fake_imgs = np.round(scale_back(fake_imgs))
            binary_real_imgs = np.round(scale_back(real_imgs))
            intersection = np.count_nonzero(np.multiply(binary_fake_imgs, binary_real_imgs))
            union = np.count_nonzero(np.add(binary_fake_imgs, binary_real_imgs))
            iou = intersection/union
            total_train_l1_loss += l1_loss
            total_train_iou += iou

        val_count = 0
        for bid, batch in enumerate(val_iter):
            val_count += 1
            fake_imgs, real_imgs, _, _, l1_loss, _ = self.generate_fake_samples(batch)
            binary_fake_imgs = np.round(scale_back(fake_imgs))
            binary_real_imgs = np.round(scale_back(real_imgs))
            intersection = np.count_nonzero(np.multiply(binary_fake_imgs, binary_real_imgs))
            union = np.count_nonzero(np.add(binary_fake_imgs, binary_real_imgs))
            iou = intersection/union
            total_val_l1_loss += l1_loss
            total_val_iou += iou

        return total_train_l1_loss/train_count, total_train_iou/train_count, total_val_l1_loss/val_count, total_val_iou/val_count

    def save_lists(self):
        model_id, _ = self.get_model_id_and_dir()
        model_lists_dir = os.path.join(self.lists_dir, model_id)
        if not os.path.exists(model_lists_dir):
            os.makedirs(model_lists_dir)

        counter_path = os.path.join(model_lists_dir, "counter_list.obj")
        train_path = os.path.join(model_lists_dir, "train_lists.obj")
        val_path = os.path.join(model_lists_dir, "val_lists.obj")
        with open(train_path, 'wb') as t:
            pickle.dump(self.train_l1_loss_list, t)
            pickle.dump(self.train_iou_list, t)
        with open(val_path, 'wb') as v:
            pickle.dump(self.val_l1_loss_list, v)
            pickle.dump(self.val_iou_list, v)
        with open(counter_path, 'wb') as c:
            pickle.dump(self.counter_list, c)

    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, model_dir, save_dir):
        source_provider = InjectDataProvider(source_obj)

        source_iter = source_provider.get_random_iter(self.batch_size)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        for source_imgs in source_iter:
            fake_imgs = self.generate_fake_samples(source_imgs)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True,
              freeze_encoder=False, fine_tune=None, sample_steps=50, checkpoint_steps=500):
        g_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        input_handle, loss_handle, _, summary_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered")

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        g_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_handle.g_loss, var_list=g_vars)
        tf.global_variables_initializer().run()
        real_data = input_handle.real_data

        # filter by one type of labels
        data_provider = TrainDataProvider(self.data_dir, filter_by=fine_tune)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_infinite_val_iter(self.batch_size)
        train_val_batch_iter = data_provider.get_infinite_train_iter(self.batch_size)

        saver = tf.train.Saver(max_to_keep=3)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            epoch_start_time = time.time()
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                batch_start_time = time.time()
                counter += 1
                batch_images = batch

                # Optimize G
                _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                feed_dict={
                                                    real_data: batch_images,
                                                    learning_rate: current_lr
                                                })

                # magic move to Optimize G again
                # according to https://github.com/carpedm20/DCGAN-tensorflow
                # collect all the losses along the way
                _, batch_g_loss, \
                const_loss, l1_loss, l2_edge_loss, g_summary = self.sess.run([g_optimizer,
                                                                         loss_handle.g_loss,
                                                                         loss_handle.const_loss,
                                                                         loss_handle.l1_loss,
                                                                         loss_handle.l2_edge_loss,
                                                                         summary_handle.g_merged],
                                                                        feed_dict={
                                                                            real_data: batch_images,
                                                                            learning_rate: current_lr,
                                                                        })
                batch_passed = time.time() - batch_start_time
                epoch_passed = time.time() - epoch_start_time
                total_passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] total time: %4.4f, epoch time: %4.4f, batch time: %4.4f, g_loss: %.5f, " + \
                             "const_loss: %.5f, l1_loss: %.5f, l2_edge_loss: %.5f"
                print(log_format % (ei, bid, total_batches, total_passed, epoch_passed, batch_passed, batch_g_loss,
                                    const_loss, l1_loss, l2_edge_loss))
                summary_writer.add_summary(g_summary, counter)

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    self.sample_model(val_batch_iter, ei, counter)
                    self.sample_model(train_val_batch_iter, ei, counter, is_train_data=True)

                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint(saver, counter)

            # output results for the special characters
            special_val_iter = data_provider.get_val_spec_iter(self.batch_size)
            self.sample_model(special_val_iter, ei, counter, is_special_data=True)

            # validate the current model states with train and val data
            train_l1_loss, train_iou, val_l1_loss, val_iou = self.validate_model(data_provider)
            self.counter_list.append(ei)
            self.train_l1_loss_list.append(train_l1_loss)
            self.train_iou_list.append(train_iou)
            self.val_l1_loss_list.append(val_l1_loss)
            self.val_iou_list.append(val_iou)

        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)

        # Save loss and iou lists
        print("Saving counter, loss and iou lists:")
        self.save_lists()