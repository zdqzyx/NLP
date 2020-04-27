# -*- coding: utf-8 -*-
# @Time : 2020/4/24 16:24
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

import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imp_by_tensorflow2.TextTransformerEncoder.modeling import TextTransformerEncoder








