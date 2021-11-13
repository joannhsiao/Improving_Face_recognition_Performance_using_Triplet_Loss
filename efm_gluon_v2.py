#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import mxnet as mx
from mxnet import autograd, gluon, init
from mxnet.gluon import nn
import logging
import datetime
import numpy as np
import os
import sys
import copy
import random
import csv
import time
from scipy import spatial
import matplotlib.pyplot as plt
from lightcnn import *


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
 
def cosine_dist(anc, pos, neg, batch_size):
    pos_dist = []
    neg_dist = []
    for i in range(batch_size):
        pos_dist.append(mx.nd.dot(anc[i], mx.nd.transpose(pos[i])) / (mx.nd.norm(anc[i]) * mx.nd.norm(pos[i])))
        neg_dist.append(mx.nd.dot(anc[i], mx.nd.transpose(neg[i])) / (mx.nd.norm(anc[i]) * mx.nd.norm(neg[i])))
    #diff = mx.nd.array(neg_dist) - mx.nd.array(pos_dist)
        
    return pos_dist, neg_dist

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
        b, c, h, w = dshape
        self.provide_data = [('data', (3, b, c, h, w))]
        self.provide_label = [('label', (self.batch_size, 1))]

    def make_pairs(self, batch_data, pos_img, batch_size):
        dataset = []
        label = []
        batchs = batch_data.data[0]
        labels = batch_data.label[0]
        for i in range(batch_size):
            j = random.randint(0, batch_size-1)
            while int(labels[j].asscalar()) == int(labels[i].asscalar()):
                j = random.randint(0, batch_size-1)
            # dataset: [[anc1, pos1, neg1], [anc2, pos2, neg2], ...]
            dataset += [[batchs[i].asnumpy(), pos_img[int(labels[i].asscalar())].asnumpy(), batchs[j].asnumpy()]]
            label += [labels[i].asscalar(), labels[i].asscalar(), labels[j].asscalar()]
        print(len(dataset))
        return mx.nd.array(dataset), mx.nd.array(label)
            
    def __iter__(self):
        #print('begin...')
        for i in range(self.length):
            batch_data = self.data_iter.next()
            batchs, labels = self.make_pairs(batch_data, self.pos_img, self.batch_size)
            data_all = [mx.nd.array(batchs)]
            label_all = [mx.nd.array(labels)]
            data_names = ['data']
            label_names = ['label']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            #print('load success!!!')

            yield data_batch

    def reset(self):
        self.data_iter.reset()
        pass

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

def draw_figure(epoch, cnt, acc_tr, acc_te):
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('accuracy of each epoch')
    plt.grid(True)
    plt.xlim(0, epoch)
    plt.ylim(0, 100)
    plt.plot(cnt, acc_tr, 'r-', label='training')
    plt.plot(cnt, acc_te, 'b-', label='testing')
    plt.legend()
    plt.savefig('train_acc.jpg')


""" Data & Parameter setting """
root = sys.argv[1]

Training_Rec_path = os.path.join(root, 'train.rec')   # Aug_train data
Testing_Rec_path = os.path.join(root, 'test.rec')     # Org_test data
Training_lst_path = os.path.join(root, 'train.lst')
Testing_lst_path = os.path.join(root, 'test.lst')


# To count the number of training and validation set
with open(Training_lst_path, "r") as file:
    data = file.readlines()
    Training_IMG_number = len(data)

with open(Testing_lst_path, "r") as file:
        data = file.readlines()
        Testing_IMG_number = len(data)

print("Totoal number of training samples = ", Training_IMG_number, flush=True)
print("Totoal number of testing samples = ", Testing_IMG_number, flush=True)


Training_IMG_size = 64
Training_IMG_channel = 1    # gray
batch_size = 64
Training_IMG_classes = 8398
epoch_size = Training_IMG_number / batch_size
dshape = (batch_size, Training_IMG_channel, Training_IMG_size, Training_IMG_size)


""" path for saving output """
Log_save_dir = 'try2_efm_light_29_134/log/'
ensure_dir(Log_save_dir)
Model_save_dir = 'try2_efm_light_29_134/model/'
ensure_dir(Model_save_dir)
Model_save_name = 'try2_efm_light_29'


""" logging info: values, training time, accuracy """
logging.basicConfig(filename=Log_save_dir + Model_save_name + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log", level=logging.INFO)
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)


""" Data Iterator """
train_dataiter = mx.io.ImageRecordIter(path_imgrec=Training_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size=batch_size, preprocess_threads=14)
train_dataiter.provide_data[0] = mx.io.DataDesc('data', (batch_size, Training_IMG_channel, Training_IMG_size, Training_IMG_size), np.float32)
test_dataiter = mx.io.ImageRecordIter(path_imgrec=Testing_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size=batch_size, preprocess_threads=14)


""" Triplet pairs """
print('defining positive image...', flush=True)
# Store a positive image for each identities
pos_img_train = define_pos(train_dataiter, int(epoch_size), batch_size)
pos_img_test = define_pos(test_dataiter, int(Testing_IMG_number/batch_size), batch_size)

train_dataiter.reset()
test_dataiter.reset()

print('making training pairs...', flush=True)
data_train = DataIter(train_dataiter, int(epoch_size), pos_img_train, batch_size, dshape)
print('making testing pairs...', flush=True)
data_test = DataIter(test_dataiter, int(Testing_IMG_number/batch_size), pos_img_test, batch_size, dshape)


""" main process """
lr = 0.00024
#group2ctx = {'stage1' : mx.gpu(0), 'stage2' : mx.gpu(1)}
devs = mx.gpu(0)
alpha = 0.1
MARGIN = 0.2

print('build network...', flush=True)
net = LightCNN_29(Training_IMG_classes)
net.initialize(ctx=devs, init=init.Xavier())
net.hybridize()
triplet_loss = gluon.loss.TripletLoss(margin=MARGIN)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
schedule = mx.lr_scheduler.FactorScheduler(step=int(epoch_size * 6), factor=0.88, stop_factor_lr=5e-15)
op = mx.optimizer.Adam(learning_rate=lr, lr_scheduler=schedule, wd=0.00001)
trainer = gluon.Trainer(net.collect_params(), op)
#trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':lr, 'wd':0.00001})


print('start training...', flush=True)
cnt = []
acc_tr = []
acc_te = []
for epoch in range(280):
    train_loss, train_acc, valid_loss, valid_acc = 0., 0., 0., 0.
    tic = time.time()
    """ training loop """
    for batch in data_train:
        anc = batch.data[0][:, 0].as_in_context(devs)
        pos = batch.data[0][:, 1].as_in_context(devs)
        neg = batch.data[0][:, 2].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        anc.attach_grad()
        pos.attach_grad()
        neg.attach_grad()
        with mx.autograd.record():              # record gradient of error
            output, a_fc = net(anc)
            _, p_fc = net(pos)
            _, n_fc = net(neg)
            TL_loss = triplet_loss(a_fc/mx.nd.norm(a_fc), p_fc/mx.nd.norm(p_fc), n_fc/mx.nd.norm(n_fc))
            id_loss = softmax_cross_entropy(output, label[0:batch_size])
            loss = id_loss + alpha * TL_loss
        loss.backward()                         # backpropagation
        trainer.step(batch_size, ignore_stale_grad=True)
        train_loss += mx.nd.mean(loss).asscalar()
        train_acc += acc(output, label[0:batch_size])
        #train_acc += np.sum(np.where(loss.asnumpy() == 0, 1, 0))

        """ save positive distance and negative distance to csv """
        pos_dist, neg_dist = cosine_dist(a_fc, p_fc, n_fc, batch_size)
        with open("cosine_similarity.csv", "a+", newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            for v in range(batch_size):
                csvwriter.writerow([pos_dist[v].asscalar(), neg_dist[v].asscalar()])

    acc_tr.append((train_acc/(Training_IMG_number/batch_size)) * 100)

    """ validation loop """
    for batch in data_test:
        anc = batch.data[0][:, 0].as_in_context(devs)
        pos = batch.data[0][:, 1].as_in_context(devs)
        neg = batch.data[0][:, 2].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        with mx.autograd.record():
            output, a_fc = net(anc)
            _, p_fc = net(pos)
            _, n_fc = net(neg)
        TL_loss = triplet_loss(a_fc/mx.nd.norm(a_fc), p_fc/mx.nd.norm(p_fc), n_fc/mx.nd.norm(n_fc))
        id_loss = softmax_cross_entropy(output, label[0:batch_size])
        loss = id_loss + alpha * TL_loss
        valid_loss += mx.nd.mean(loss).asscalar()
        #valid_acc += np.sum(np.where(loss.asnumpy() == 0, 1, 0))
        valid_acc += acc(output, label[0:batch_size])

    acc_te.append((valid_acc/(Testing_IMG_number/batch_size)) * 100)
    cnt.append(epoch)

    data_train.reset()
    data_test.reset()

    """ save parameters for each epoch """
    paramfile = "efm_res-%04d.params" %(epoch)
    net.save_parameters(paramfile)

    print("Epoch {}: train loss {:g}, train acc {:g}, valid loss: {:g}, valid acc {:g}, in {:.1f} sec".format(
            epoch, train_loss/(Training_IMG_number/batch_size), train_acc/(Training_IMG_number/batch_size), 
            valid_loss/(Testing_IMG_number/batch_size), valid_acc/(Testing_IMG_number/batch_size), time.time()-tic), flush=True)


""" figure of accuracy """
draw_figure(epoch, cnt, acc_tr, acc_te)
