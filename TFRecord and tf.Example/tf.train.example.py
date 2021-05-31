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


# Creating a tf.train.Example message

# first we have to create four type features eg: boolean, int, string, float

# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)
#_>print(len(feature1))

# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)


# next we can encode features by methods of: _bytes_feature, _float_feature, _int64_feature 
def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


# convert example features with above function
# This is an example observation from the dataset.

example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
#_>print(serialized_example) # here we got long line encoded result


# To decode the message
example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto)