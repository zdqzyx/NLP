# -*- coding: utf-8 -*-
# @Time : 2020/4/20 14:44
# @Author : zdqzyx
# @File : textcnn.py
# @Software: PyCharm

import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model

class TextCNN(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes=[1,2,3],
                 kernel_regularizer=None,
                 last_activation='softmax'
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
        :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
        :param class_num:
        :param last_activation:
        '''
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        # self.max_features = max_features
        # self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.conv1s = []
        self.avgpools = []
        for kernel_size in kernel_sizes:
            self.conv1s.append(Conv1D(filters=128, kernel_size=kernel_size, activation='relu', kernel_regularizer=kernel_regularizer))
            self.avgpools.append(GlobalMaxPooling1D())
        self.classifier = Dense(class_num, activation=last_activation, )

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](emb) # (batch_size, maxlen-kernel_size+1, filters)
            c = self.avgpools[i](c) # # (batch_size, filters)
            conv1s.append(c)
        x = Concatenate()(conv1s) # (batch_size, len(self.kernel_sizes)*filters)
        output = self.classifier(x)
        return output

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)




