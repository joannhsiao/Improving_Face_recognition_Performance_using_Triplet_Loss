#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import mxnet as mx
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from lightcnn import *
from efm_gluon import *

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

def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

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

Rec_path = os.path.join(root, 'lfw_data.rec')
lst_path = os.path.join(root, 'lfw_data.lst')

IMG_number = 0
file = open(lst_path, "r")
for linen in file:
    IMG_number += 1
file.close()

print("Totoal number of training samples = ", IMG_number)


IMG_size = 128
IMG_channel = 1    # gray
batch_size = 32
epoch_size = IMG_number / batch_size
dshape = (batch_size, IMG_channel, IMG_size, IMG_size)
#epoch_size = IMG_number / batch_size
print("epoch_size = %d" %(epoch_size))

""" Data Iterator """
dataiter = mx.io.ImageRecordIter(path_imgrec=Rec_path, shuffle=False, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(IMG_channel, IMG_size, IMG_size), batch_size=batch_size, preprocess_threads=14)

'''
""" projection """
test_data = []
test_label = []
for batch in dataiter:
    for i in range(batch_size):
        test_data.append(batch.data[0][i].asnumpy())
        test_label.append(batch.label[0][i].asscalar())
test_data, test_label = transform(mx.nd.array(test_data), mx.nd.array(test_label))
tb_projector(test_data, test_label, os.path.join(ROOT_DIR, 'try2_efm_light_29_134', 'logs', 'origin'))

dataiter.reset()
'''

"""  process of feature extraction """
devs = mx.gpu(0)

print('loading network...')
symbol = mx.sym.load("%s/EFM_RES.json" % sys.argv[2])
inputs = mx.sym.var("data")
#id_out = symbol.get_internals()['fc2_output']
fc_out = symbol.get_internals()['concat29_output']
#sym = mx.symbol.Group([id_out, fc_out])
model = mx.gluon.SymbolBlock(fc_out, inputs)
model.collect_params().load("%s/EFM_RES.params" % sys.argv[2], ctx=devs)
#net.load_params("%s/feature.params" % sys.argv[2], ctx=devs)


print('Generating feature vector...', flush=True)
tic = time.time()
for epoch in range(1):
    for batch in dataiter:
    	data = batch.data[0].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        fc = model(data)
        """ save feature vectors """
        with open("feature_vector_test.csv", "a+", newline='') as csvfile:
            for v in range(batch_size):
                anc_list = (anc[v] / mx.nd.norm(anc[v])).asnumpy().tolist()
                for ele in anc_list:
                    csvfile.write("{},".format(ele))
                csvfile.write("\n")
        
        """ save labels """
        with open("label_test.csv", "a+", newline='') as csvfile:
            for v in range(batch_size):
                csvfile.write("{}".format(label[v].asscalar()))
                csvfile.write("\n")


""" test for cosine similarity """
feature_dim = 342
batch_size = 1024
epoch_size = IMG_number / batch_size
Training_IMG_classes = 8398
dshape = (batch_size, feature_dim)

test_dataiter = mx.io.CSVIter(data_csv='feature_vector_test.csv', data_shape=(1, feature_dim), label_csv='label_test.csv', label_shape=(1,), batch_size=batch_size)

print('defining positive image...', flush=True)
# Store a positive image for each identities
pos_img = define_pos(test_dataiter, int(epoch_size), batch_size)

test_dataiter.reset()

print('making pairs...', flush=True)
data_test = DataIter(test_dataiter, int(epoch_size), pos_img, batch_size, dshape)

print('build network...', flush=True)
net = nn.Sequential()
net.add(nn.Dense(feature_dim, use_bias=False))
net.collect_params().load("%s/fc_efm_res-0299.params" % sys.argv[2], ctx=devs)

triplet_loss = gluon.loss.TripletLoss(margin=MARGIN)

print('start testing...', flush=True)
test_loss = 0.
tic = time.time()
for epoch in range(1):
    for batch in data_test:
        data = batch.data[0]
        n_data = []
        for i in range(batch_size*2):
            n_data.append((data[i] / mx.nd.norm(data[i])).asnumpy())
        n_data = mx.nd.array(n_data).as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        neg = []
        fc = net(n_data)
        """ triplet pairs"""
        anc = fc[0: batch_size]
        pos = fc[batch_size:batch_size*2]
        for i in range(batch_size):
            j = random.randint(0, batch_size*2 - 1)
            while int(label[j].asscalar()) == int(label[i].asscalar()):
                j = random.randint(0, batch_size*2 - 1)
            neg.append(fc[j].asnumpy())
        neg = mx.nd.array(neg).as_in_context(devs)
        """ loss function """
        loss = triplet_loss(anc/mx.nd.norm(anc), pos/mx.nd.norm(pos), neg/mx.nd.norm(neg))
        valid_loss += mx.nd.mean(loss).asscalar()

        """ save positive and negative distance to csv """
        pos_dist, neg_dist = cosine_dist(anc, pos, neg, batch_size)
        with open("cosine_similarity.csv", "a+", newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            for v in range(batch_size):
                csvwriter.writerow([pos_dist[v].asscalar(), neg_dist[v].asscalar()])


""" Log metrics to std output """
print("Test acc %g, in %.1f sec" % (test_acc/(IMG_number/batch_size), time.time()-tic))

'''
""" projection """
test_pred, test_label = transform(mx.nd.array(pred), mx.nd.array(test_label))
tb_projector(test_pred, test_label, os.path.join(ROOT_DIR, 'try2_efm_light_29_134', 'logs', 'prediction'))
'''

