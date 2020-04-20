# -*- coding: utf-8 -*-
# @Time : 2020/4/20 14:16
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fasttext import FastText

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    # >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


class ModelHepler:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        print('Bulid Model...')
        self.create_model()

    def create_model(self):
        model = FastText(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                         last_activation='softmax')
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        model.build_graph(input_shape=(None, maxlen))
        model.summary()
        self.model =  model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='logs\\FastText-epoch-5', checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, mode='max')
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
                                             period=2,
                                             )
            callback_list.append(cp_callback)
        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs/FastText-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=2,
                  callbacks=self.callback_list,
                  validation_data=(x_val, y_val))

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        # 创建一个新的模型实例
        # model = self.create_model()
        # 加载以前保存的权重
        self.model.load_weights(latest)

# ================  params =========================
ngram_range = 1
class_num = 2
maxlen = 400
embedding_dims = 100
epochs = 10
batch_size = 128
max_features = 5000

MODEL_NAME = 'FastText-epoch-10-emb-100-maxpooling'

use_early_stop=True
tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_path = 'save_model_dir\\'+MODEL_NAME+'\\cp-{epoch:04d}.ckpt'
#  ====================================================================

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
print('Pad sequences (samples x time)...')
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size
                           )
model_hepler.get_callback(use_early_stop=use_early_stop, tensorboard_log_dir=tensorboard_log_dir, checkpoint_path=checkpoint_path)
model_hepler.fit(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)
print('Test...')
result = model_hepler.model.predict(x_test)
test_score = model_hepler.model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print("test loss:", test_score[0], "test accuracy", test_score[1])



model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size
                           )
model_hepler.load_model(checkpoint_path=checkpoint_path)
# 重新评估模型
loss, acc = model_hepler.model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))









