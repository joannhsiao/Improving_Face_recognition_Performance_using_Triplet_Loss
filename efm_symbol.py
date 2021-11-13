#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import mxnet as mx
import memonger
import logging
import datetime
import numpy as np
import os
import sys
import copy
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def res_block(data, num_r, layer):
    num_r1 = int(num_r * (2. / 3.))

    slice_r = mx.symbol.SliceChannel(data=data, num_outputs=3, name=('slice%s_res' % layer))
    efm_r_max1 = mx.symbol.maximum(slice_r[0], slice_r[1])
    efm_r_min1 = mx.symbol.minimum(slice_r[0], slice_r[1])
    efm_r_max2 = mx.symbol.maximum(slice_r[2], efm_r_max1)
    efm_r_min2 = mx.symbol.minimum(slice_r[2], efm_r_min1)
    efm_r = mx.symbol.Concat(efm_r_max2, efm_r_min2)
    
    conv_r = mx.symbol.Convolution(data=efm_r, num_filter=num_r, kernel=(3, 3), name=('conv%s_res' % layer), pad=(1, 1))
    
    slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=3, name=('slice%s_res' % layer))
    efm_r_max1 = mx.symbol.maximum(slice_r[0], slice_r[1])
    efm_r_min1 = mx.symbol.minimum(slice_r[0], slice_r[1])
    efm_r_max2 = mx.symbol.maximum(slice_r[2], efm_r_max1)
    efm_r_min2 = mx.symbol.minimum(slice_r[2], efm_r_min1)
    efm_r = mx.symbol.Concat(efm_r_max2, efm_r_min2)
    
    conv_r1 = mx.symbol.Convolution(data=efm_r, num_filter=num_r1, kernel=(3, 3), name=('conv%s_res_r' % layer), pad=(1, 1))
    conv_r0 = data + conv_r1
    
    return conv_r0

def group(data, num_r, num, kernel, stride, pad, layer, tar_num=0):
    if num_r > 0:
        if num_r % 3 == 0:
            if tar_num >= 1:
                res = res_block(data, num_r, layer)
            if tar_num >= 2:
                for x in range(1, tar_num):
                    res = res_block(res, num_r, layer + str(x))
            conv_r2 = mx.symbol.Convolution(data=res, num_filter=num_r, kernel=(1, 1), name=('conv%s_r' % layer))
            slice_r = mx.symbol.SliceChannel(data=conv_r2, num_outputs=3, name=('slice%s_r' % layer))
            efm_r_max1 = mx.symbol.maximum(slice_r[0], slice_r[1])
            efm_r_min1 = mx.symbol.minimum(slice_r[0], slice_r[1])
            efm_r_max2 = mx.symbol.maximum(slice_r[2], efm_r_max1)
            efm_r_min2 = mx.symbol.minimum(slice_r[2], efm_r_min1)
            mfm_r = mx.symbol.Concat(efm_r_max2, efm_r_min2)
        else:
            conv_r = mx.symbol.Convolution(data=data, num_filter=num_r, kernel=(1, 1), name=('conv%s_r' % layer))
            slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
            mfm_r = mx.symbol.maximum(slice_r[0], slice_r[1])
        conv = mx.symbol.Convolution(data=mfm_r, kernel=kernel, stride=stride, pad= pad, num_filter=num, name=('conv%s' % layer))
    else:
        conv = mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
    if num % 3 == 0:
        slice = mx.symbol.SliceChannel(data=conv, num_outputs=3, name=('slice%s' % layer))
        mfm_max1 = mx.symbol.maximum(slice[0], slice[1])
        mfm_max2 = mx.symbol.maximum(mfm_max1, slice[2])
        mfm_min1 = mx.symbol.minimum(slice[0], slice[1])
        mfm_min2 = mx.symbol.minimum(mfm_min1, slice[2])
        mfm = mx.symbol.Concat(mfm_max2, mfm_min2)
    else:
        slice = mx.symbol.SliceChannel(data=conv, num_outputs=2, name=('slice%s' % layer))
        mfm = mx.symbol.maximum(slice[0], slice[1])
    pool = mx.symbol.Pooling(data=mfm, pool_type="max", kernel=(2, 2), stride=(2, 2), name=('pool%s' % layer))
    return pool

def multi_gpu(data, classes):
    with mx.AttrScope(ctx_group='stage1'):
        #data = mx.symbol.Variable(name = "data")
        pool1 = group(data, 0, 99, (5, 5), (1, 1), (2, 2), str(1))          # res: 0
        pool2 = group(pool1, 99, 198, (3, 3), (1, 1), (1, 1), str(2), 1)    # res: 1
        pool3 = group(pool2, 198, 387, (3, 3), (1, 1), (1, 1), str(3), 2)   # res: 2
        pool4 = group(pool3, 387, 261, (3, 3), (1, 1), (1, 1), str(4), 3)   # res: 3
    
    set_stage1 = set(pool4.list_arguments())
    
    with mx.AttrScope(ctx_group='stage2'):
        pool5 = group(pool4, 261, 261, (3, 3), (1, 1), (1, 1), str(5), 4)   # res: 4
        flatten = mx.symbol.Flatten(data=pool5)
        fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=513, name="fc1")
        
        slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=3, name="slice_fc1")
        efm_fc_max1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
        efm_fc_min1 = mx.symbol.minimum(slice_fc1[0], slice_fc1[1])
        efm_fc_max2 = mx.symbol.maximum(slice_fc1[2], efm_fc_max1)
        efm_fc_min2 = mx.symbol.minimum(slice_fc1[2], efm_fc_min1)
        efm_fc1 = mx.symbol.Concat(efm_fc_max2, efm_fc_min2)

        drop1 = mx.symbol.Dropout(data=efm_fc1, p=0.7, name="drop1")
        fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=classes, name="fc2")
        softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    
    set_stage2 = set(softmax.list_arguments()) - set_stage1
    group2ctx = {'stage1' : mx.gpu(0), 'stage2' : mx.gpu(0)}
    
    return softmax, fc1

def get_net(classes, margin=0.2):
    data = mx.sym.Variable('data')
    
    output, fc = multi_gpu(data, classes)
    '''
    digraph = mx.viz.plot_network(output, save_format='png')
    digraph.view()
    '''

    loss = mx.sym.MakeLoss(triplet_loss)
    multi_out = mx.symbol.Group([output, fc])
    return multi_out

def cosine_dist(a, b):
    a1 = mx.nd.expand_dims(a, axis=1)   # (1,28,28) -> (1,1,28,28)
    b1 = mx.nd.expand_dims(b, axis=0)   # (1,28,28) -> (1,1,28,28)
    #d = mx.nd.batch_dot(a1, b1).squeeze()   # (1,1,28,28) -> (28, 28)
    #a_norm = mx.nd.sqrt(mx.nd.sum(mx.nd.square(a1)))    # scalar
    #b_norm = mx.nd.sqrt(mx.nd.sum(mx.nd.square(b1)))    # scalar
    #cos = d / (a_norm * b_norm) # (28,28)
    a2 = mx.nd.flatten(a).asnumpy()
    b2 = mx.nd.flatten(b).asnumpy()
    cos = mx.nd.array(1 - spatial.distance.cosine(a2, b2))
    #dist = mx.nd.sqrt(mx.nd.sum(mx.nd.square(b1 - a1)))  #scalar
    return cos

# create a dataset of positive image for each identities
def define_pos(data_iter, length, batch_size):
    pos_img = {}
    for epoch in range(length):
        for batch in data_iter:
            for i in range(batch_size):
                if int(batch.label[0][i].asscalar()) not in pos_img:
                    pos_img[int(batch.label[0][i].asscalar())] = copy.copy(batch.data[0][i])
    # pos_img: {1: img1, 2: img2, ...}
    return pos_img

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self, data_iter, length, pos_img, batch_size, dshape):
        super(DataIter, self).__init__()
        
        self.batch_size = batch_size
        self.length = length
        self.data_iter = data_iter
        self.pos_img = pos_img
        self.dshape = dshape
        self.provide_data = [('data', self.dshape)]
        self.provide_label = [('label', (self.batch_size, 1))]
        
        self.batch, self.label = self.make_pairs(self.data_iter, self.pos_img, self.batch_size)

    def make_pairs(self, data_iter, pos_img, batch_size):
        dataset = []
        labels = []
        for batch in data_iter:
            for i in range(batch_size / 2):
                # dataset: [[anc1, pos1], [anc2, pos2], ...]
                dataset += [[batch.data[0][i].asnumpy(), pos_img[int(batch.label[0][i].asscalar())].asnumpy()]]
                labels += [batch.label[0][i].asnumpy()]
        print(len(dataset), len(labels))
        print(len(labels[0][0][0]))
        return dataset, labels
        
    def __iter__(self):
        print('begin...')
        start = 0
        for i in range(self.length):
            data = []
            label = []
            for j in range(start, start + self.batch_size / 2):
                data.append(self.batch[0][j])
                data.append(self.batch[1][j])
                label.append(self.label[j])
                label.append(self.label[j])
            
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['label']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            print('load success!!!')
            start += (self.batch_size / 2)

            yield data_batch

    def reset(self):
        self.data_iter.reset()
        pass

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('Auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)

class triplet_loss(mx.metric.EvalMetric):
    def __init__(self):
        super(triplet_loss, self).__init__('Triplet_loss')

    def pair(self, labels, preds):


    def update(self, labels, preds):
        self.pair(labels, preds)
        self.pos = preds
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)

def transform(data, label):
    return data.astype(np.float32) / 255., label.astype(np.float32)

def tb_projector(X_test, y_test, log_dir):
    metadata = os.path.join(log_dir, 'metadata.tsv')
    images = tf.Variable(X_test)
    with open(metadata, 'w') as metadata_file: # write to metadata
        for row in y_test:
            metadata_file.write('%d\n' % row)
    with tf.Session() as sess:
        saver = tf.train.Saver([images])  # store data to matrix
        sess.run(images.initializer)  # image intialization
        saver.save(sess, os.path.join(log_dir, 'images.ckpt'))  # save image
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()     # add embedding vector
        embedding.tensor_name = images.name
        embedding.metadata_path = metadata
        projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)  # visualization


""" Data & Parameter setting """
root = sys.argv[1]

Training_Rec_path = os.path.join(root, 'train.rec')   # Aug_train data
Testing_Rec_path = os.path.join(root, 'test.rec')     # Org_test data
Training_lst_path = os.path.join(root, 'train.lst')
Testing_lst_path = os.path.join(root, 'test.lst')

names = []
for fn in os.listdir(root):
    names.append(root + '/' + fn)   # incomplete
Training_IMG_classes = len(names)

# To count the number of training and validation set
with open(Training_lst_path, "r") as file:
    data = file.readlines()
    Training_IMG_number = len(data)

with open(Testing_lst_path, "r") as file:
        data = file.readlines()
        Testing_IMG_number = len(data)

print("Totoal number of training samples = ", Training_IMG_number)
print("Totoal number of testing samples = ", Testing_IMG_number)


#Training_IMG_number = 410804
Training_IMG_size = 64
Training_IMG_channel = 1    # RGB
#Training_IMG_classes = 2
batch_size = 64
epoch_size = Training_IMG_number / batch_size
dshape = (batch_size, Training_IMG_channel, Training_IMG_size, Training_IMG_size)   # (batch, 3, 128, 128)


""" path for saving output """
Log_save_dir = 'try2_efm_light_29_134/log/'
ensure_dir(Log_save_dir)
Model_save_dir = 'try2_efm_light_29_134/model/'
ensure_dir(Model_save_dir)
Model_save_name = 'try2_efm_light_29'


""" logging info: values, training time, accuracy """
logging.basicConfig(filename = Log_save_dir + Model_save_name + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log", level=logging.INFO)
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)


""" Data Iterator """
train_dataiter = mx.io.ImageRecordIter(path_imgrec=Training_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size=batch_size, preprocess_threads=14)
train_dataiter.provide_data[0] = mx.io.DataDesc('data', (batch_size, Training_IMG_channel, Training_IMG_size, Training_IMG_size), np.float32)
test_dataiter = mx.io.ImageRecordIter(path_imgrec=Testing_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size=batch_size, preprocess_threads=14)

'''
""" projection """
test_data = []
test_label = []
for batch in test_dataiter:
    for i in range(batch_size):
        test_data.append(batch.data[0][i])
        test_label.append(batch.label[0][i])
test_data, test_label = transform(mx.nd.array(test_data), mx.nd.array(test_label))
tb_projector(test_data, test_label, os.path.join(ROOT_DIR, 'logs', 'origin'))

test_dataiter.reset()
'''

""" Triplet pairs """
print('defining positive image...')
# Store a positive image for each identities
pos_img_train = define_pos(train_dataiter, int(epoch_size), batch_size)
pos_img_test = define_pos(test_dataiter, int(Testing_IMG_number / batch_size), batch_size)

train_dataiter.reset()
test_dataiter.reset()

print('making training pairs...')
data_train = DataIter(train_dataiter, int(epoch_size), pos_img_train, batch_size, dshape)
print('making testing pairs...')
data_test = DataIter(test_dataiter, int(Testing_IMG_number / batch_size), pos_img_test, batch_size, dshape)


""" main process """
lr = 0.00024
#group2ctx = {'stage1' : mx.gpu(0), 'stage2' : mx.gpu(1)}
devs = [mx.gpu(0)]

print('build model...')
new_sym = get_net(Training_IMG_classes, margin=0.2)
#net_mem_planned = memonger.search_plan(new_sym, data=dshape)

mod = mx.mod.Module(symbol=new_sym[0], data_names=('data', ), label_names=('label', ), context=devs)
mod.bind(data_shapes=data_train.provide_data, label_shapes=data_train.provide_label)
mod.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))
#mod.set_params(new_args, aux_params, allow_missing=True)

kv = mx.kvstore.create('local')
op = mx.optimizer.create('adam', rescale_grad=(1.0 / batch_size), lr_scheduler=mx.lr_scheduler.FactorScheduler(step=int(epoch_size * 6), factor=0.88, stop_factor_lr=5e-15), learning_rate=lr, beta1=0.9, wd=0.00001)
checkpoint = mx.callback.do_checkpoint(Model_save_dir + Model_save_name)

#mod = mx.model.FeedForward(ctx=devs, symbol=new_sym, num_epoch=280, initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
metric = Auc()
mod.fit(data_train, eval_data=data_test, eval_metric=metric, num_epoch=280, optimizer=op, batch_end_callback=mx.callback.Speedometer(batch_size, 100), kvstore=kv, epoch_end_callback=checkpoint)

for epoch in range(epoch_size):
    data_train.reset()
    metric.reset()
    for batch in data_train:
        mod.forward(batch, is_train=True)
        mod.update_metric(metric, batch.label)
        mod.backward()
        mod.update()
    print("epoch %d: Training %s").format(epoch, metric.get())

print(mod.tojson())
mod.save('model.json')


""" projection """
_, test_pred, _, _, _ = mod.predict(X=test_dataiter, num_batch=batch_size)
tb_projector(test_pred.asnumpy(), test_label, os.path.join(ROOT_DIR, 'logs', 'triplet'))


""" calculate cosine distance """

