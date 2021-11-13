# -*- coding:utf-8 -*-
from mxnet.gluon import nn
import mxnet as mx
import gluoncv

class efm(nn.HybridBlock):
    def __init__(self, num_filter, num_filter1, kernel_size, stride, padding, efm_type, **kwargs):
        super(efm, self).__init__(**kwargs)
        self.num_filter = num_filter
        self.num_filter1 = num_filter1
        self.efm_type = efm_type
        
        if num_filter > 0:
            self.conv_op_1 = nn.Conv2D(channels=self.num_filter, kernel_size=(1, 1))
        self.conv_op_2 = nn.Conv2D(channels=self.num_filter1, kernel_size=kernel_size, strides=stride, padding=padding)

    def hybrid_forward(self, F, x):
        # efm_type == 1 or 0
        # efm_type = 1, conv + ele-max + conv + ele-max; otherwise, conv + ele-max
        if self.efm_type == 1:
            self.conv_1 = self.conv_op_1(x)
            self.slice_conv1 = F.SliceChannel(data=self.conv_1, num_outputs=3, axis=1)
            self.efm_conv1_max1 = F.maximum(self.slice_conv1[0], self.slice_conv1[1])
            self.efm_conv1_max2 = F.maximum(self.efm_conv1_max1, self.slice_conv1[2])
            self.efm_conv1_min1 = F.minimum(self.slice_conv1[0], self.slice_conv1[1])
            self.efm_conv1_min2 = F.minimum(self.efm_conv1_min1, self.slice_conv1[2])
            self.efm_conv1 = F.Concat(self.efm_conv1_max2, self.efm_conv1_min2)
            self.conv_2 = self.conv_op_2(self.efm_conv1)
        else:
            self.conv_2 = self.conv_op_2(x)

        self.slice_conv2 = F.SliceChannel(data=self.conv_2, num_outputs=3, axis=1)
        self.efm_conv2_max1 = F.maximum(self.slice_conv2[0], self.slice_conv2[1])
        self.efm_conv2_max2 = F.maximum(self.efm_conv2_max1, self.slice_conv2[2])
        self.efm_conv2_min1 = F.minimum(self.slice_conv2[0], self.slice_conv2[1])
        self.efm_conv2_min2 = F.minimum(self.efm_conv2_min1, self.slice_conv2[2])
        self.efm_conv = F.Concat(self.efm_conv2_max2, self.efm_conv2_min2)

        return self.efm_conv

class res_block(nn.HybridBlock):
    def __init__(self, num_blocks, num_filter, **kwargs):
        super(res_block, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_filter1 = int(num_filter * (2. / 3.))
        
        self.conv_op_1 = nn.Conv2D(channels=num_filter, kernel_size=(3, 3), padding=(1, 1))
        self.conv_op_2 = nn.Conv2D(channels=self.num_filter1, kernel_size=(3, 3), padding=(1, 1))

    def hybrid_forward(self, F, x):
        in_data = x
        for i in range(self.num_blocks):
            self.slice_conv1 = F.SliceChannel(data=in_data, num_outputs=3, axis=1)
            self.efm_conv1_max1 = F.maximum(self.slice_conv1[0], self.slice_conv1[1])
            self.efm_conv1_max2 = F.maximum(self.efm_conv1_max1, self.slice_conv1[2])
            self.efm_conv1_min1 = F.minimum(self.slice_conv1[0], self.slice_conv1[1])
            self.efm_conv1_min2 = F.minimum(self.efm_conv1_min1, self.slice_conv1[2])
            self.efm_conv1 = F.Concat(self.efm_conv1_max2, self.efm_conv1_min2)
            self.conv_1 = self.conv_op_1(self.efm_conv1)

            self.slice_conv2 = F.SliceChannel(data=self.conv_1, num_outputs=3, axis=1)
            self.efm_conv2_max1 = F.maximum(self.slice_conv2[0], self.slice_conv2[1])
            self.efm_conv2_max2 = F.maximum(self.efm_conv2_max1, self.slice_conv2[2])
            self.efm_conv2_min1 = F.minimum(self.slice_conv2[0], self.slice_conv2[1])
            self.efm_conv2_min2 = F.minimum(self.efm_conv2_min1, self.slice_conv2[2])
            self.efm_conv = F.Concat(self.efm_conv2_max2, self.efm_conv2_min2)
            self.conv_2 = self.conv_op_2(self.efm_conv)

            in_data = self.conv_2 + in_data
        
        return in_data

class LightCNN_29(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(LightCNN_29, self).__init__(**kwargs)

        num_blocks = [1, 2, 3, 4]
        with self.name_scope():
            self.conv_net = nn.HybridSequential()
            self.conv_net.add(
                # group 1: no res_block
                efm(num_filter=0, num_filter1=99, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), efm_type=0),
                nn.MaxPool2D(pool_size=2, strides=2),

                # group 2
                res_block(num_blocks=num_blocks[0], num_filter=99),
                #efm(num_filter=198, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), efm_type=0),
                efm(num_filter=99, num_filter1=198, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), efm_type=1),
                nn.MaxPool2D(pool_size=2, strides=2),

                # group 3
                res_block(num_blocks=num_blocks[1], num_filter=198),
                #efm(num_filter=387, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), efm_type=0),
                efm(num_filter=198, num_filter1=387, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), efm_type=1),
                nn.MaxPool2D(pool_size=2, strides=2),

                # group 4
                res_block(num_blocks=num_blocks[2], num_filter=387),
                #efm(num_filter=261, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), efm_type=0),
                efm(num_filter=387, num_filter1=261, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), efm_type=1),
                nn.MaxPool2D(pool_size=2, strides=2),

                # group 5
                res_block(num_blocks=num_blocks[3], num_filter=261),
                #efm(num_filter=261, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), efm_type=0),
                efm(num_filter=261, num_filter1=261, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), efm_type=1),
                nn.MaxPool2D(pool_size=2, strides=2),

                nn.Flatten(),
                # fc1 
                nn.Dense(1026))
            
            self.fc1 = nn.HybridSequential()
            self.fc1.add(nn.BatchNorm())

            self.fc2 = nn.HybridSequential()
            self.fc2.add(nn.Dropout(.7), 
                        nn.Dense(num_classes))
    
    def hybrid_forward(self, F, x):
        fc1 = self.conv_net(x)

        self.slice_fc1 = F.SliceChannel(data=fc1, num_outputs=3, axis=1)
        self.efm_fc1_max1 = F.maximum(self.slice_fc1[0], self.slice_fc1[1])
        self.efm_fc1_max2 = F.maximum(self.efm_fc1_max1, self.slice_fc1[2])
        self.efm_fc1_min1 = F.minimum(self.slice_fc1[0], self.slice_fc1[1])
        self.efm_fc1_min2 = F.minimum(self.efm_fc1_min1, self.slice_fc1[2])
        self.efm_fc1 = F.Concat(self.efm_fc1_max2, self.efm_fc1_min2)

        fc1_out = self.fc1(self.efm_fc1)
        out = self.fc2(self.efm_fc1)

        return out, fc1_out

'''
if __name__ == '__main__':
    net = LightCNN_29(100)
    params = net.collect_params()
    print(params)
    gluoncv.utils.viz.plot_network(net)
'''

