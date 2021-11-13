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

def group(data, num_r, num, kernel, stride, pad, layer, tar_num = 0):
    if num_r > 0:
        if num_r % 3 == 0:
            if tar_num >= 1:
                res = res_block(data, num_r, layer)
            if tar_num >= 2:
                for x in xrange(1, tar_num):
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

def mutli_gpu(classes):
    with mx.AttrScope(ctx_group='stage1'):
        data = mx.symbol.Variable(name = "data")
        pool1 = group(data, 0, 99, (5,5), (1,1), (2,2), str(1))
        pool2 = group(pool1, 99, 198, (3,3), (1,1), (1,1), str(2), 1)
        pool3 = group(pool2, 198, 387, (3,3), (1,1), (1,1), str(3), 2)
        pool4 = group(pool3, 387, 261, (3,3), (1,1), (1,1), str(4), 3)
    
    set_stage1 = set(pool4.list_arguments())
    
    with mx.AttrScope(ctx_group='stage2'):
        pool5 = group(pool4, 261, 261, (3, 3), (1, 1), (1, 1), str(5), 4)
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
    group2ctx = {'stage1' : mx.gpu(0), 'stage2' : mx.gpu(1)}
    
    return softmax


""" Data & Parameter setting """
Training_Rec_path = '/home/mnxnet-cv307/mxnet_model/SynDataAugmentation/trainImg.rec'   # Aug_train data
Testing_Rec_path = '/home/mnxnet-cv307/mxnet_model/SynpowerTraningData/testImg.rec'     # Org_test data

Training_IMG_number = 410804
Training_IMG_size = 128
Training_IMG_channel = 3    # RGB
Training_IMG_classes = 2
batch_size = 100
epoch_size = Training_IMG_number / batch_size
dshape = (batch_size, Training_IMG_channel, Training_IMG_size, Training_IMG_size)   # (100, 3, 128, 128)

lr = 0.00024
#group2ctx = {'stage1' : mx.gpu(0), 'stage2' : mx.gpu(1)}
devs = [mx.gpu(0), mx.gpu(1)]

""" path for saving output """
Log_save_dir = '/home/mnxnet-cv307/mxnet_model/FaceId_model/Augemation_traindata/try2_efm_light_29_134/log/'
ensure_dir(Log_save_dir)
Model_save_dir = '/home/mnxnet-cv307/mxnet_model/FaceId_model/Augemation_traindata/try2_efm_light_29_134/model/'
ensure_dir(Model_save_dir)
Model_save_name = 'try2_efm_light_29'


""" logging info: values, training time, accuracy """
logging.basicConfig(filename = Log_save_dir + Model_save_name + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log", level=logging.INFO)
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)


""" Data Iterator """
train_dataiter = mx.io.ImageRecordIter(path_imgrec=Training_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size = batch_size, preprocess_threads = 14)
train_dataiter.provide_data[0] = mx.io.DataDesc('data', (batch_size, Training_IMG_channel, Training_IMG_size, Training_IMG_size), np.float32)

test_dataiter = mx.io.ImageRecordIter(path_imgrec=Testing_Rec_path, shuffle=True, scale=1./255, rand_crop=True, rand_mirror=True, data_shape=(Training_IMG_channel, Training_IMG_size, Training_IMG_size), batch_size = batch_size, preprocess_threads = 14)


""" main process"""
# use mutli_gpu to create a framework of Model, 
# use memonger.search_plan to optimize the usage of the memory
# put framework on GPU (mx.mod.Module), and bind the training data (mod.bind)，
# setting the position of shared data (mx.kvstore), the optimizer (mx.optimizer) 
# and the path for saving parameters (mx.callback), 
# mod.fit for training
""""""
new_sym = mutli_gpu(Training_IMG_classes)
net_mem_planned = memonger.search_plan(new_sym, data=dshape)

mod = mx.mod.Module(symbol=net_mem_planned, context=devs)
mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)

mod.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))
#mod.set_params(new_args, aux_params, allow_missing=True)
kv = mx.kvstore.create('local')
op = mx.optimizer.create('adam', rescale_grad=(1.0 / batch_size), lr_scheduler=mx.lr_scheduler.FactorScheduler(step=int(epoch_size * 6), factor=0.88, stop_factor_lr=5e-15), learning_rate=lr, beta1=0.9, wd=0.00001)
checkpoint = mx.callback.do_checkpoint(Model_save_dir + Model_save_name)

mod.fit(train_dataiter, test_dataiter, eval_metric='acc', num_epoch=280, batch_end_callback=mx.callback.Speedometer(batch_size, 100), kvstore=kv, optimizer=op, epoch_end_callback=checkpoint)

