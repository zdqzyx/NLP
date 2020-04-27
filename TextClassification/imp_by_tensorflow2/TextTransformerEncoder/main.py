# -*- coding: utf-8 -*-
# @Time : 2020/4/21 11:44
# @Author : zdqzyx
# @File : main.py
# @Software: PyCharm


# ===================== set random  ===========================
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)

# =============================================================

import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from TextClassification.imp_by_tensorflow2.TextTransformerEncoder.modeling import TextTransformerEncoder

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


class ModelHepler:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size, num_layers,
                 num_heads,
                 dff,
                 pe_input,):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.pe_input = pe_input

    def create_model(self):
        input_seq = tf.keras.layers.Input((self.maxlen, ), dtype=tf.int64, name='inputs_seq')
        input_mask = tf.keras.layers.Input((1,1,self.maxlen), dtype=tf.float32, name='inputs_mask')
        internal_model = TextTransformerEncoder(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                                       num_layers=self.num_layers,
                                       num_heads=self.num_heads,
                                       dff=self.dff,
                                       pe_input=self.pe_input,
                         last_activation='softmax',
                          # dense_size=[128]
                          )
        logits = internal_model(input_seq, enc_padding_mask=input_mask)
        logits = tf.keras.layers.Lambda(lambda x: x, name="logits",
                                        dtype=tf.float32)(logits)
        model = tf.keras.Model([input_seq, input_mask], logits)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        # model.build_graph(input_shape=(None, self.maxlen))
        model.summary()
        self.model =  model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='logs\\FastText-epoch-5', checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
            callback_list.append(early_stopping)
        if checkpoint_path is not None:
            # save model
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            # 创建一个保存模型权重的回调
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                             monitor='val_accuracy',
                                             mode='max',
                                             save_best_only=True,
                                             save_weights_only=True,
                                             verbose=1,
                                             period=1,
                                             )
            callback_list.append(cp_callback)
        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs/FastText-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val, x_train_mask, x_val_mask):
        print('Bulid Model...')
        self.create_model()
        print('Train...')
        self.model.fit([x_train, x_train_mask], y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=2,
                  callbacks=self.callback_list,
                  validation_data=((x_val, x_val_mask), y_val))

    def load_model(self, checkpoint_path):
        self.create_model()
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        # 创建一个新的模型实例
        # model = self.create_model()
        # 加载以前保存的权重
        self.model.load_weights(latest)

# ================  params =========================
class_num = 2
maxlen = 400
embedding_dims = 200
epochs = 20
batch_size = 128
max_features = 5000

num_layers = 2
num_heads = 4
dff = 128
pe_input = max_features

MODEL_NAME = 'TextTransformerEncoder-epoch-20-emb-200'

use_early_stop=True
tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_path = 'save_model_dir\\'+MODEL_NAME+'\\cp-{epoch:04d}.ckpt'
#  ====================================================================

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('Pad sequences (samples x time)...')
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_train_mask = create_padding_mask(x_train)
x_test_mask = create_padding_mask(x_test)

print('x_train_mask shape:', x_train_mask.shape)

#
model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        dff=dff,
                        pe_input=pe_input
                           )
model_hepler.get_callback(use_early_stop=use_early_stop, tensorboard_log_dir=tensorboard_log_dir, checkpoint_path=checkpoint_path)
model_hepler.fit(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, x_train_mask=x_train_mask, x_val_mask=x_test_mask)


print('Test predict...')
result = model_hepler.model.predict((x_test, x_test_mask))
print('Test evaluate...')
test_score = model_hepler.model.evaluate((x_test, x_test_mask), y_test,
                            batch_size=batch_size)
print("test loss:", test_score[0], "test accuracy", test_score[1])


print('Restored Model...')
model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size,
                           num_layers=num_layers,
                           num_heads=num_heads,
                           dff=dff,
                           pe_input=pe_input
                           )
model_hepler.load_model(checkpoint_path=checkpoint_path)
# 重新评估模型  0.8790
loss, acc = model_hepler.model.evaluate((x_test, x_test_mask), y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))