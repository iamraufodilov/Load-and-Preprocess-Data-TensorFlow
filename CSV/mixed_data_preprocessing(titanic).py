# load libraries
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
#load dataset
titanic = pd.read_csv('G:/rauf/STEPBYSTEP/Tutorial/TensorFlow Core/Load and Preprocess Data TensorFlow/CSV/train.csv')
#_>print(titanic.head())


# get label and feature
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')


# 
inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

#_>print(inputs)


# conctente numeric collumns and normalize them
numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

#_>print(all_numeric_inputs)
preprocessed_inputs = [all_numeric_inputs]


#convert string features to ctegorical value by one hot encoding
for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)


# concatenate processed inputsand inputs
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)


# convert data to dictionary
titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}


# look at first example from our dict
features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
print(titanic_preprocessing(features_dict)) # good, now our all features are become numeric and as well normalized


# create model
def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
  return model

titanic_model = titanic_model(titanic_preprocessing, inputs)


# train the model
titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
we load titanic dataset which is messy
the dataset has numeric and string walues
we normalize numeric walues
we categorize string vlues
and concatenate categorized string and normalized numeric values as final dictionary
then train our model
'''