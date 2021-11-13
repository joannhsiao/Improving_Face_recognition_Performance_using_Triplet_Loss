#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import mxnet as mx
import numpy as np
import os
import sys
import time
import copy
import random
import csv
import matplotlib.pyplot as plt


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
        labels = []
        batch = data_iter.data[0]
        label = data_iter.label[0]
        #for batch in data_iter:
        for i in range(batch_size):
            # dataset: [[anc1, pos1], [anc2, pos2], ...]
            dataset += [[batch[i].asnumpy(), pos_img[int(label[i].asscalar())].asnumpy()]]
            labels += [label[i].asscalar()]
        #print(len(dataset))
        return mx.nd.array(dataset), mx.nd.array(labels)
        
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


with open("train_id.csv", "r") as file:
    data = file.readlines()
    IMG_number = len(data)
print("Totoal number of samples = ", IMG_number, flush=True)

""" test for cosine similarity """
feature_dim = 342
batch_size = 4096 * 4
epoch_size = IMG_number / batch_size
#Training_IMG_classes = 79078
dshape = (batch_size, feature_dim)

print("epoch_size: {}".format(epoch_size), flush=True)

test_dataiter = mx.io.CSVIter(data_csv='train_img.csv', data_shape=(1, feature_dim), label_csv='train_id.csv', label_shape=(1,), batch_size=batch_size)

print('defining positive image...', flush=True)
# Store a positive image for each identities
pos_img = define_pos(test_dataiter, int(epoch_size), batch_size)

test_dataiter.reset()

print('making pairs...', flush=True)
data_test = DataIter(test_dataiter, int(epoch_size), pos_img, batch_size, dshape)


""" parameter setting """
#lr = 0.00024
#group2ctx = {'stage1' : mx.gpu(0), 'stage2' : mx.gpu(1)}
devs = mx.gpu(0)
#MARGIN = 0.2
'''
print('build network...', flush=True)
net = nn.Sequential()
net.add(nn.Dense(342, use_bias=False))
triplet_loss = gluon.loss.TripletLoss(margin=MARGIN)
net.initialize(init=init.Xavier(), ctx=devs)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.00001})
'''

print('start testing...', flush=True)
#test_loss = 0.
#tic = time.time()
for epoch in range(1):
    cnt = 0
    for batch in data_test:
        tic = time.time()
        data = batch.data[0].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        """ data normalization """
        n_data = []
        for i in range(batch_size*2):
            n_data.append((data[i] / mx.nd.norm(data[i])).asnumpy())
        n_data = mx.nd.array(n_data).as_in_context(devs)
        """ weight matrix """
        #Wnx = net(n_data)
        """ triplet pairs"""
        anc = n_data[0: batch_size]
        pos = n_data[batch_size:batch_size*2]
        neg = []
        for i in range(batch_size):
            j = random.randint(0, batch_size*2 - 1)
            while int(label[j].asscalar()) == int(label[i].asscalar()):
                j = random.randint(0, batch_size*2 - 1)
            neg.append(n_data[j].asnumpy())
        neg = mx.nd.array(neg).as_in_context(devs)
        """ loss function """
        #loss = triplet_loss(anc, pos, neg)
        #test_loss += mx.nd.mean(loss).asscalar()

        """ save positive and negative distance to csv """
        pos_dist, neg_dist = cosine_dist(anc, pos, neg, batch_size)
        with open("cosine_similarity.csv", "a+", newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            for v in range(batch_size):
                csvwriter.writerow([pos_dist[v].asscalar(), neg_dist[v].asscalar()])

        cnt += 1
        print("[batch {}]: in {:.1f} sec".format(cnt, time.time()-tic), flush=True)


