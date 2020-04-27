# -*- coding: utf-8 -*-
# @Time : 2020/4/24 21:07
# @Author : zdqzyx
# @File : custom_main.py
# @Software: PyCharm

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from Transformer.imp_by_tensorflow20_custom.modeling import Transformer

def create_model(params, is_train):
    """Creates transformer model."""
    with tf.name_scope("model"):
        if is_train:
            inputs = tf.keras.layers.Input((None,), dtype='int64', name="inputs")
            targets = tf.keras.layers.Input((None,), dtype='int64', name="targets")
            internal_model = Transformer(params,)
            logits = internal_model([inputs, targets], training=is_train)
            # logits = tf.keras.layers.Lambda(lambda x: x, name="logits", dtype=tf.float32)(logits)
            model = tf.keras.Model([inputs, targets], logits)
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
            )
            # model.build_graph(input_shape=(None, self.maxlen))
            # model.summary()
            return model
        else:
            inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
            internal_model = Transformer(params, name="transformer_v2")
            ret = internal_model([inputs], training=is_train)
            outputs, scores = ret["outputs"], ret["scores"]
            return tf.keras.Model(inputs, [outputs, scores])




params = {
        'num_layers':2,
        'd_model':512,
        'num_heads' :8,
        'dff' :2048,
        'input_vocab_size' : 9,
        'target_vocab_size' :9,
        'pe_input' :100,
        'pe_target' : 100,
        'rate':0.1,
        'tar_max_len':200
        }
model = create_model(params, is_train=True)

inp = np.array([[1,2,3,4,5,6,7,8,0,0,],
                [1,2,3,4,5,6,7,8,0,0,]])

# inp = tf.convert_to_tensor(inp, dtype=tf.int64)

tar = np.array([[1,2,3,4,5,6,7,8,0,0,],
                [1,2,3,4,5,6,7,8,0,0,]])
# tar = tf.convert_to_tensor(tar, dtype=tf.int64)

inp = tf.random.uniform((64, 62), maxval=params['input_vocab_size'], dtype=tf.int64)
tar = tf.random.uniform((64, 26), maxval=params['target_vocab_size'], dtype=tf.int64)

print(inp[0])

model.fit(x=[inp, tar], y=tar)

