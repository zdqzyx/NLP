# -*- coding: utf-8 -*-
# @Time : 2020/4/22 16:28
# @Author : zdqzyx
# @File : modeling.py
# @Software: PyCharm

import tensorflow as tf
import  numpy as np

def get_angles(pos, i, d_model):
    '''
    :param pos:单词在句子的位置
    :param i:单词在词表里的位置
    :param d_model:词向量维度大小
    :return:
    '''
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    '''
    :param position: 最大的position
    :param d_model: 词向量维度大小
    :return: [1, 最大position个数，词向量维度大小] 最后和embedding矩阵相加
    '''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


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
    # (batch_size, num_heads, seq_len_q, d ) dot (batch_size, num_heads, d, seq_ken_k) = (batch_size, num_heads,, seq_len_q, seq_len)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 缩放matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        # (batch_size, num_heads,, seq_len_q, seq_len) + (batch_size, 1,, 1, seq_len)
        scaled_attention_logits += (mask * -1e9)

    # softmax归一化权重 (batch_size, num_heads, seq_len)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # seq_len_q个位置分别对应v上的加权求和
    # (batch_size, num_heads, seq_len) dot (batch_size, num_heads, d_v) = (batch_size, num_heads, seq_len_q, d_v)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (d_model > num_heads) and (d_model % num_heads == 0)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.qw = tf.keras.layers.Dense(d_model)
        self.kw = tf.keras.layers.Dense(d_model)
        self.vw = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

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
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=(0, 2, 1, 3)) # (batch_size, seq_len_q, num_heads, depth_v)
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    '''Encoder block
    包括两个子层：1.多头注意力（有填充遮挡）2.点式前馈网络（Point wise feed forward networks）。
    out1 = BatchNormalization( x +（MultiHeadAttention(x, x, x)=>dropout）)
    out2 = BatchNormalization( out1 + (ffn(out1) => dropout) )
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layer_norm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x+attn_output) # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1+ffn_output) # (batch_size, input_seq_len, d_model)
        return out2



class Encoder(tf.keras.layers.Layer):
    '''
    输入嵌入（Input Embedding）
    位置编码（Positional Encoding）
    N 个编码器层（encoder layers）
    输入经过嵌入（embedding）后，该嵌入与位置编码相加。该加法结果的输出是编码器层的输入。编码器的输出是解码器的输入。
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layer = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # mask == (batch_size, 1, 1, seq_len) 后面会自动广播成（batch_size, num_heads, seq_len_q, seq_len_k)
        # x.shape == (batch_size, seq_len)
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layer[i](x, training, mask)
        return  x #(batch_size, input_seq_len, d_model)

class TextTransformerEncoder(tf.keras.Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 num_layers,
                 num_heads,
                 dff,
                 pe_input,
                 last_activation = 'softmax',
                 rate=0.1):
        super(TextTransformerEncoder, self).__init__()
        self.maxlen = maxlen
        self.embedding_dims = embedding_dims
        self.encoder = Encoder(num_layers, embedding_dims, num_heads, dff, max_features, pe_input, rate)
        # self.flat = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(class_num, activation=last_activation)

    def call(self, inp, training=False, enc_padding_mask=None):
        # mask == (batch_size, 1, 1, seq_len) 后面会自动广播成（batch_size, num_heads, seq_len, seq_len)
        # (batch_size, inp_seq_len, d_model)
        print('========= : ', training)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        # enc_output = tf.reshape(enc_output, (-1, self.maxlen*self.embedding_dims))
        # enc_output = self.flat(enc_output)
        enc_output = tf.reduce_mean(enc_output, axis=1)
        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output

    def build_graph(self, input_shapes):
        input_shape, _ = input_shapes
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)



if __name__=='__main__':
    class_num = 2
    maxlen = 400
    embedding_dims = 400
    epochs = 10
    batch_size = 128
    max_features = 5000

    num_layers = 2
    num_heads = 8
    dff = 2048
    pe_input = 10000
    model = TextTransformerEncoder(
                        maxlen=maxlen,
                        max_features=max_features,
                        embedding_dims=embedding_dims,
                        class_num=class_num,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        dff=dff,
                        pe_input=pe_input
                        )
    # model.build_graph(input_shapes=(None, maxlen))
    # model.summary()
    temp_input = tf.random.uniform((64, maxlen))

    # fn_out= model(temp_input, training=False, enc_padding_mask=None)
    fn_out= model([temp_input, None], training=False)

    print(fn_out.shape)












