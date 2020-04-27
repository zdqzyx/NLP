# -*- coding: utf-8 -*-
# @Time : 2020/4/21 13:55
# @Author : zdqzyx
# @File : attention.py
# @Software: PyCharm

import tensorflow as tf
tf.keras.layers.Masking


def scaled_dot_product_attention(q, k, v, mask=None):
    '''计算attention
    q,k,v的第一维度必须相同
    q,k的最后一维必须相同
    k,v在倒数第二的维度需要相同, seq_len_k = seq_len_q=seq_len。
    参数:
    q: 请求的形状 == (..., seq_len_q, d)
    k: 主键的形状 == (..., seq_len, d)
    v: 数值的形状 == (..., seq_len, d_v)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len)。默认为None。
    返回值:
    输出，注意力权重
    '''
    # (n, seq_len_q, d ) dot (n, d, seq_ken_k) = (n, seq_len_q, seq_len)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 缩放matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax归一化权重 (n, seq_len_q, seq_len)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # seq_len_q个位置分别对应v上的加权求和
    # (n, seq_len_q, seq_len) dot (n, seq_len, d_v) = (n, seq_len_q, d_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (d_model > num_heads) and (d_model % num_heads == 0), '{}, {}'.format(d_model, num_heads)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.qw = tf.keras.layers.Dense(d_model)
        self.kw = tf.keras.layers.Dense(d_model)
        self.vw = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

        self.scaled_dot_product_attention = scaled_dot_product_attention

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))


    def call(self, v, k, q, mask=None):
        # v = inputs
        batch_size = tf.shape(q)[0]

        q = self.qw(q)  # (batch_size, seq_len_q, d_model)
        k = self.kw(k)  # (batch_size, seq_len, d_model)
        v = self.vw(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth_v)

        # scaled_attention, (batch_size, num_heads, seq_len_q, depth_v)
        # attention_weights, (batch_size, num_heads, seq_len_q, seq_len)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=(0, 2, 1, 3)) # (batch_size, seq_len_q, num_heads, depth_v)
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output


if __name__=='__main__':
    x = tf.ones((2, 5, 10))
    att = MultiHeadAttention(10, 2)
    y = att(x, x, x)
    print(y.shape)
    print(y)



