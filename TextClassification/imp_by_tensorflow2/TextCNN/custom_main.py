# -*- coding: utf-8 -*-
# @Time : 2020/4/20 19:50
# @Author : zdqzyx
# @File : custom_main.py
# @Software: PyCharm

# ===================== set random  ===========================
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)
# =============================================================


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textcnn import TextCNN


class_num = 2
maxlen = 400
max_features = 300
embedding_dims = 100
epochs = 10
batch_size = 6

max_features = 5000

print(('max_features is : ', max_features))

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


print('Build model...')
model = TextCNN(maxlen=maxlen,
                 max_features=max_features,
                 embedding_dims=embedding_dims,
                 class_num=class_num,
                 kernel_sizes=[2,3,5],
                 kernel_regularizer=None,
                 last_activation='softmax')

# 为训练选择优化器与损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

#选择衡量指标来度量模型的损失值（loss）和准确率（accuracy）。这些指标在 epoch 上累积值，然后打印出整体结果。
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# 使用 tf.GradientTape 来训练模型：
@tf.function
def train_step(x, labels):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

# 测试模型：
@tf.function
def test_step(x, labels):
  predictions = model(x)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

def inference(x):
    predictions = model(x)
    return np.argmax(predictions, axis=-1)


print('Train...')
EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for x, labels in train_ds:
        train_step(x, labels)

    for x, test_labels in test_ds:
        test_step(x, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

print('Test...')
pred = np.array([])
for x, test_labels in test_ds:
    pred = np.append(pred, inference(x))
print("pred is : ", pred)