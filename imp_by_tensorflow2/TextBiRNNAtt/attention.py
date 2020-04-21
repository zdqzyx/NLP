# -*- coding: utf-8 -*-
# @Time : 2020/4/21 13:55
# @Author : zdqzyx
# @File : attention.py
# @Software: PyCharm

from tensorflow.keras import  initializers,regularizers,constraints
from  tensorflow.keras.layers import Layer
import tensorflow as tf

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs
                 ):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        super(Attention, self).__init__()
        self.bias = bias
        self.init = initializers.get('glorot_uniform')

    def build(self, input_shape):
        '''
        :param input_shape:
        :return:
        '''
        self.output_dim = input_shape[-1]
        self.W = self.add_weight(
                                 name='{}_W'.format(self.name),
                                 shape=(input_shape[2], 1),
                                 initializer=self.init,
                                 trainable=True
                                 )
        if self.bias:
            self.b = self.add_weight(
                                     name='{}_b'.format(self.name),
                                     shape=(input_shape[1], 1),
                                     initializer='zero',
                                     trainable=True
                                     )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        # (N, step, d), (d, 1)  ==>   (N, step, 1)
        e = tf.matmul(inputs, self.W, )
        if self.bias:
            e += self.b
        e = tf.tanh(e)
        a = tf.nn.softmax(e, axis=1)
        # (N, step, d) (N, step, 1) ====> (N, step, d)
        c = inputs*a
        # (N, d)
        c = tf.reduce_sum(c, axis=1)
        return c

    def get_config(self):
        return {'units': self.output_dim}


if __name__=='__main__':
    x = tf.ones((2, 5, 10))
    att = Attention()
    y = att(x)
    print(y.shape)
    print(y)
    print(att.get_config())


