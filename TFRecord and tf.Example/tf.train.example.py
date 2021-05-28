# load librries
import tensorflow as tf

import numpy as np
import IPython.display as display


# lets create some functions to convert standart tensorflow to compatible tf.train.features

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


#lets check those functions
#_>print(_bytes_feature(b'test_string'))
#_>print(_bytes_feature(u'test_bytes'.encode('utf-8')))

#_>print(_float_feature(np.exp(1)))

#_>print(_int64_feature(True))
#_>print(_int64_feature(1))


# all proto messages can be serialized as binary-string
feature = _float_feature(np.exp(1))

#_>print(feature.SerializeToString()) # here you can see our float number becomes binary sting


#