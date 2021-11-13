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
        self.provide_data = [('data', dshape)]
        self.provide_label = [('label', (self.batch_size, 1))]
        #self.batchs, self.labels = self.make_pairs(self.data_iter, pos_img, self.batch_size)
        
    def make_pairs(self, data_iter, pos_img, batch_size):
        dataset = []
        label = []
        batchs = data_iter.data[0]
        labels = data_iter.label[0]
        #for batch in data_iter:
        for i in range(batch_size):
            # dataset: [[anc1, pos1], [anc2, pos2], ...]
            dataset += [[batchs[i].asnumpy(), pos_img[int(labels[i].asscalar())].asnumpy()]]
            label += [labels[i].asscalar()]
        #print(len(dataset))
        return mx.nd.array(dataset), mx.nd.array(label)
        
    def __iter__(self):
        #print('begin...')
        #start = 0
        for i in range(self.length):
            batch_data = self.data_iter.next()
            batchs, labels = self.make_pairs(batch_data, self.pos_img, self.batch_size)
            data = []
            label = []
            for j in range(self.batch_size):
                data.append(batchs[j][0].asnumpy())
                label.append(labels[j].asscalar())
            for j in range(self.batch_size):
                data.append(batchs[j][1].asnumpy())
                label.append(labels[j].asscalar())
            
            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['label']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            #print('load success!!!')
            #start += self.batch_size
            yield data_batch

    def reset(self):
        self.data_iter.reset()
        pass

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()


""" Data & Parameter setting """

# To count the number of training and validation set
with open("train_id.csv", "r") as file:
    data = file.readlines()
    Training_IMG_number = len(data)

with open("test_id.csv", "r") as file:
        data = file.readlines()
        Testing_IMG_number = len(data)

print("Totoal number of training samples = ", Training_IMG_number, flush=True)
print("Totoal number of testing samples = ", Testing_IMG_number, flush=True)

feature_dim = 342
batch_size = 4096 * 4
#Training_IMG_classes = 79078
epoch_size = Training_IMG_number / batch_size
dshape = (batch_size, feature_dim)


""" path for saving output """
Log_save_dir = 'try2_efm_light_29/log/'
ensure_dir(Log_save_dir)
Model_save_dir = 'try2_efm_light_29/model/'
ensure_dir(Model_save_dir)
Model_save_name = 'try2_efm_light_29'


""" logging info: values, training time, accuracy """
logging.basicConfig(filename=Log_save_dir + Model_save_name + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log", level=logging.INFO)
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)


""" Data Iterator """
train_dataiter = mx.io.CSVIter(data_csv='train_img.csv', data_shape=(1, feature_dim), label_csv='train_id.csv', label_shape=(1,), batch_size=batch_size)
test_dataiter = mx.io.CSVIter(data_csv='test_img.csv', data_shape=(1, feature_dim), label_csv='test_id.csv', label_shape=(1,), batch_size=batch_size)


""" Triplet pairs """
print('defining positive image...', flush=True)
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
MARGIN = 0.5

print('build network...', flush=True)
net = nn.Sequential()
net.add(nn.Dense(128, use_bias=False))

triplet_loss = gluon.loss.TripletLoss(margin=MARGIN)
net.initialize(init=init.Xavier(), ctx=devs)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.00001})


print('start training...', flush=True)
for epoch in range(300):
    train_loss, valid_loss = 0., 0.
    tic = time.time()
    """ training loop """
    for batch in data_train:
        data = batch.data[0].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        neg = []
        with mx.autograd.record():              # record gradient of error
            Wnx = net(data)
            """ triplet pairs """
            anc = Wnx[0: batch_size]
            pos = Wnx[batch_size:batch_size*2]
            for i in range(batch_size):
                j = random.randint(0, batch_size - 1)
                while int(label[j].asscalar()) == int(label[i].asscalar()):
                    j = random.randint(0, batch_size - 1)
                neg.append(Wnx[j].asnumpy())
            neg = mx.nd.array(neg).as_in_context(devs)
            
            """ loss function """
            loss = triplet_loss(anc, pos, neg)
        loss.backward()                         # backpropagation
        trainer.step(batch_size)
        train_loss += mx.nd.mean(loss).asscalar()
        #train_acc += np.sum(np.where(loss.asnumpy() == 0, 1, 0))

        """ save positive and negative distance to csv """
        pos_dist, neg_dist = cosine_dist(anc, pos, neg, batch_size)
        with open("cosine_similarity.csv", "a+", newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            for v in range(batch_size):
                csvwriter.writerow([pos_dist[v].asscalar(), neg_dist[v].asscalar()])


    """ validation loop """
    for batch in data_test:
        data = batch.data[0].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        neg = []
        Wnx = net(data)
        """ triplet pairs"""
        anc = Wnx[0: batch_size]
        pos = Wnx[batch_size:batch_size*2]
        for i in range(batch_size):
            j = random.randint(0, batch_size*2 - 1)
            while int(label[j].asscalar()) == int(label[i].asscalar()):
                j = random.randint(0, batch_size*2 - 1)
            neg.append(Wnx[j].asnumpy())
        neg = mx.nd.array(neg).as_in_context(devs)
        
        """ loss function """
        loss = triplet_loss(anc, pos, neg)
        valid_loss += mx.nd.mean(loss).asscalar()
        #valid_acc += np.sum(np.where(loss.asnumpy() == 0, 1, 0))

    data_train.reset()
    data_test.reset()

    """ save parameters for each epoch """
    paramfile = "fc_efm_res-%04d.params" %(epoch)
    net.save_parameters(paramfile)

    print("Epoch {}: train loss {:g}, valid loss {:g}, in {:.1f} sec".format(
        epoch, train_loss/epoch_size, valid_loss/(Testing_IMG_number/batch_size), time.time()-tic), flush=True)

