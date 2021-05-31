#load libraries
import collections
import pathlib
import re
import string

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text


#load dataset
train_dir = 'G:/rauf/STEPBYSTEP/Data/stacjoverflow_text/train'



# lets look one example of data
sample_file = train_dir/'python/1755.txt'
with open(sample_file) as f:
  print(f.read())
