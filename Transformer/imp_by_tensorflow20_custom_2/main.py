# -*- coding: utf-8 -*-
# @Time : 2020/4/22 21:46
# @Author : zdqzyx
# @File : main.py
# @Software: PyCharm

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from Transformer.imp_by_tensorflow20_custom_2.modeling import Transformer
import os

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2

MAX_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)
def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en

train_preprocessed = (
    train_examples
    .map(tf_encode)
    .filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    .cache()
    .shuffle(BUFFER_SIZE))

val_preprocessed = (
    val_examples
    .map(tf_encode)
    .filter(filter_max_length))

train_dataset = (train_preprocessed
                 .padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))
                 .prefetch(tf.data.experimental.AUTOTUNE))


val_dataset = (val_preprocessed
               .padded_batch(BATCH_SIZE,  padded_shapes=([None], [None])))



def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    '''
    eg.
    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    temp:<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
            array([[0., 1., 1.],
                   [0., 0., 1.],
                   [0., 0., 0.]], dtype=float32)>
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)
    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)
    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1]) #(tar_seq_len, tar_seq_len)
    dec_target_padding_mask = create_padding_mask(tar) # (batch_size, 1, 1, tar_seq_len)
    # 广播机制，look_ahead_mask==>(batch_size, 1, tar_seq_len, tar_seq_len)
    # dec_target_padding_mask ==> (batch_size, 1, tar_seq_len, tar_seq_len)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


# ==============================================================
params = {
    'num_layers':4,
    'd_model':128,
    'dff':512,
    'num_heads':8,
    'input_vocab_size' :tokenizer_pt.vocab_size + 2,
    'target_vocab_size':tokenizer_en.vocab_size + 2,
    'pe_input':tokenizer_pt.vocab_size + 2,
    'pe_target':tokenizer_en.vocab_size + 2,
    'rate':0.1,
    'checkpoint_path':'./checkpoints/train',
    'checkpoint_do_delete':False
}


print('input_vocab_size is {}, target_vocab_size is {}'.format(params['input_vocab_size'], params['target_vocab_size']))


class ModelHelper:

    def __init__(self):
        self.transformer  = Transformer(params)
        # optimizer
        learning_rate = CustomSchedule(params['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # 主要为了累计一个epoch中的batch的loss，最后求平均，得到一个epoch的loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # 主要为了累计一个epoch中的batch的acc，最后求平均，得到一个epoch的acc
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


        # 检查点 params['checkpoint_path']如果不存在，则创建对应目录；如果存在，且checkpoint_do_delete=True时，则先删除目录在创建
        checkout_dir(dir_path=params['checkpoint_path'], do_delete=params.get('checkpoint_do_delete', False))
        # 检查点
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, params['checkpoint_path'], max_to_keep=5)
        # 如果检查点存在，则恢复最新的检查点。
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]
    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    @tf.function
    def test_step(self, inp, labels):
        predictions = self.predict(inp)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self, train_dataset):
        for epoch in range(params['epochs']):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)
                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,ckpt_save_path))
            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, self.train_loss.result(), self.train_accuracy.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    # 评估
    def predict(self, inp_sentence):
        start_token = [tokenizer_pt.vocab_size]
        end_token = [tokenizer_pt.vocab_size + 1]

        # 输入语句是葡萄牙语，增加开始和结束标记
        inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # 因为目标是英语，输入 transformer 的第一个词应该是
        # 英语的开始标记。
        decoder_input = [tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # 从 seq_len 维度选择最后一个词
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # 如果 predicted_id 等于结束标记，就返回结果
            if predicted_id == tokenizer_en.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights
            # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)


    def evaluation(self, x, y):
        self.test_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_step(x, y, training=False)
        template = 'Evaluation || Loss: {}, Accuracy: {}'
        print(template.format(self.test_loss.result(), self.test_accuracy.result() * 100))


def translate(model_helper, sentence):
    result, _ = model_helper.predict(sentence)
    print(result)
    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

model_helper = ModelHelper()


translate(model_helper, "este é um problema que temos que resolver.")
print ("Real translation: this is a problem we have to solve .")


translate(model_helper, "os meus vizinhos ouviram sobre esta ideia.")
print ("Real translation: and my neighboring homes heard about this idea .")


translate(model_helper, "este é o primeiro livro que eu fiz.",)
print ("Real translation: this is the first book i've ever done.")





