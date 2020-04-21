# -*- coding: utf-8 -*-
# @Time : 2020/4/21 10:47
# @Author : zdqzyx
# @File : text_birnn.py
# @Software: PyCharm

import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras import backend as K


def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn


class TextBiRNN(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='softmax',
                 dense_size=None
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param class_num:
        :param last_activation:
        '''
        super(TextBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_size = dense_size

        self.embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(layer=GRU(units=128, activation='tanh', return_sequences=True), merge_mode='concat' ) # LSTM or GRU
        # self.avepool = GlobalAveragePooling1D()
        if self.dense_size is not None:
            self.ffn = point_wise_feed_forward_network(dense_size)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNN must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        x = self.bi_rnn(emb)
        # x = self.avepool(x)
        x = tf.reduce_mean(x, axis=1)
        if self.dense_size is not None:
            x = self.ffn(x)
        output = self.classifier(x)
        return output

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)

if __name__=='__main__':
    model = TextBiRNN(maxlen=400,
                    max_features=5000,
                    embedding_dims=100,
                    class_num=2,
                    last_activation='softmax',
                    # dense_size=[128, 64],
    dense_size = None

    )
    model.build_graph(input_shape=(None, 400))
    model.summary()
    config = model.get_config()
    print(config)