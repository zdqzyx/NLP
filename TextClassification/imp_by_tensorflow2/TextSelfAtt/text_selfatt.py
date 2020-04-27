# -*- coding: utf-8 -*-
# @Time : 2020/4/21 13:50
# @Author : zdqzyx
# @File : text_selfatt.py
# @Software: PyCharm


import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, Flatten
from tensorflow.keras import Model
from TextClassification.imp_by_tensorflow2.TextSelfAtt.attention import MultiHeadAttention

def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn

class TextSelfAtt(Model):

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
        super(TextSelfAtt, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_size = dense_size

        self.embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)
        self.attention = MultiHeadAttention(d_model=embedding_dims, num_heads=4)
        self.bi_rnn = Bidirectional(layer=GRU(units=400, activation='tanh', return_sequences=True), merge_mode='ave' ) # LSTM or GRU

        if self.dense_size is not None:
            self.ffn = point_wise_feed_forward_network(dense_size)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNNAtt must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNNAtt must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        x = self.embedding(inputs)
        x = self.attention(x, x, x)
        x = self.bi_rnn(x)
        # x = tf.reduce_mean(x, axis=1)
        x = tf.reshape(x,  shape=(-1, 400*self.maxlen))
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
    model = TextSelfAtt(maxlen=400,
                        max_features=5000,
                        embedding_dims=400,
                        class_num=2,
                        last_activation='softmax',
                        dense_size=[128, 64],
                        # dense_size = None
                        )
    model.build_graph(input_shape=(None, 400))
    model.summary()
