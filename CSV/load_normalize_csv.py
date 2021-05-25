#load libraries
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


#load datset
abalone_train = pd.read_csv('G:/rauf/STEPBYSTEP/Tutorial/TensorFlow Core/Load and Preprocess Data TensorFlow/CSV/abalone_train.csv',
                            names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                                   "Viscera weight", "Shell weight", "Age"])

#_>print(abalone_train.head())


# get feature and label
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')


# transfer features into single numpy array
abalone_features = np.array(abalone_features)
#_>print(abalone_features)

'''

# create regression model to predict age
abalone_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())


# train the model
abalone_model.fit(abalone_features, abalone_labels, epochs=10) # our loss 6.5 which is not good

'''


# Basic Preprocessing


normalize = preprocessing.Normalization()
normalize.adapt(abalone_features)


norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                           optimizer = tf.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10) # here we go after normalization our loss dropped to 4.9 from 6.5


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
we load csv dataset
we train model
we normalize data
we train again
and get better result
'''