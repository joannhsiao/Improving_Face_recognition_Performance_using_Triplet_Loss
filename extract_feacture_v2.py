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
import matplotlib.pyplot as plt


def acc(output, label):
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()


""" Data & Parameter setting """
root = sys.argv[1]
Training_Rec_path = os.path.join(root, 'train.rec')
Testing_Rec_path = os.path.join(root, 'test.rec')


Training_IMG_size = 128
Training_IMG_channel = 1
batch_size = 32


""" Data Iterator """
train_dataiter = mx.io.ImageRecordIter(path_imgrec=Training_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size=batch_size, preprocess_threads=14)
train_dataiter.provide_data[0] = mx.io.DataDesc('data', (batch_size, Training_IMG_channel, Training_IMG_size, Training_IMG_size), np.float32)
test_dataiter = mx.io.ImageRecordIter(path_imgrec=Testing_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size=batch_size, preprocess_threads=14)


""" main process """
devs = mx.gpu(0)

print('Load model...', flush=True)
""" load pre-trained model"""
symbol = mx.sym.load("%s/EFM_RES.json" % sys.argv[2])
inputs = mx.sym.var("data")
id_out = symbol.get_internals()['fc2_output']
fc_out = symbol.get_internals()['concat29_output']
sym = mx.symbol.Group([id_out, fc_out])
net = mx.gluon.SymbolBlock(sym, inputs)
net.collect_params().load("%s/EFM_RES.params" % sys.argv[2], ctx=devs)


print('Processing...', flush=True)
for epoch in range(1):
    cnt = 0
    """ training loop """
    for batch in train_dataiter:
        train_loss, train_acc = 0., 0.
        tic = time.time()
        data = batch.data[0].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        output, fc = net(data)
        
        train_acc += acc(output, label)
        
        """ save feature vectors """
        with open("feature_vector_train.csv", "a+", newline='') as csvfile:
            for v in range(batch_size):
                fc_list = (fc[v] / mx.nd.norm(fc[v])).asnumpy().tolist()
                for ele in fc_list:
                    csvfile.write("{},".format(ele))
                csvfile.write("\n")
        
        """ save labels """
        with open("label_train.csv", "a+", newline='') as csvfile:
            for v in range(batch_size):
                csvfile.write("{}".format(label[v].asscalar()))
                csvfile.write("\n")

        print("[batch {}]: train acc {:g}, in {:.1f} sec".format(cnt, train_acc, time.time()-tic), flush=True)
        cnt += 1
    """ validation loop """
    cnt = 0
    for batch in test_dataiter:
        valid_loss, valid_acc = 0., 0.
        tic = time.time()
        data = batch.data[0].as_in_context(devs)
        label = batch.label[0].as_in_context(devs)
        
        output, fc = net(data)
        valid_acc += acc(output, label)

        """ save feature vectors """
        with open("feature_vector_valid.csv", "a+", newline='') as csvfile:
            for v in range(batch_size):
                fc_list = (fc[v] / mx.nd.norm(fc[v])).asnumpy().tolist()
                for ele in fc_list:
                    csvfile.write("{},".format(ele))
                csvfile.write("\n")
        
        """ save labels """
        with open("label_valid.csv", "a+", newline='') as csvfile:
            for v in range(batch_size):
                csvfile.write("{}".format(label[v].asscalar()))
                csvfile.write("\n")

        print("[batch {}]: valid acc {:g}, in {:.1f} sec".format(cnt, valid_acc, time.time()-tic), flush=True)
        cnt += 1

