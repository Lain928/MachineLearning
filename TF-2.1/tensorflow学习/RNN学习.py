import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #取消AVX2的警告

imdb = tf.keras.datasets.imdb
vocab_size = 10000
index_from = 3
(train_data, train_lab), (test_data, test_lab) = imdb.load_data(num_words=vocab_size, index_from=index_from)
print(train_data[0],train_lab[0])
print(train_data.shape,train_lab.shape)
word_index = imdb.get_word_index()
print(len(word_index))
