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


# load dataset
train_ds = tfds.load(
    'imdb_reviews',
    split='train',
    batch_size=BATCH_SIZE,
    shuffle_files=True,
    as_supervised=True)

val_ds = tfds.load(
    'imdb_reviews',
    split='train',
    batch_size=BATCH_SIZE,
    shuffle_files=True,
    as_supervised=True)


# print few examples
for review_batch, label_batch in val_ds.take(1):
  for i in range(5):
    print("Review: ", review_batch[i].numpy())
    print("Label: ", label_batch[i].numpy())


#prepare dataset

#vectorize layer
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

# Make a text-only dataset (without labels), then call adapt
train_text = train_ds.map(lambda text, labels: text)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)

# Configure datasets for performance as before
train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)


# create the model
model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=1)
model.summary()


# compile the model
model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])


# train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=3)


# evaluate the model
loss, accuracy = model.evaluate(val_ds)

print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))


# export the model
export_model = tf.keras.Sequential(
    [vectorize_layer, model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy'])


# predict unseen custom data with expert model
# 0 --> negative review
# 1 --> positive review
inputs = [
    "This is a fantastic movie.",
    "This is a bad movie.",
    "This movie was so bad that it was good.",
    "I will never say yes to watching this movie.",
]
predicted_scores = export_model.predict(inputs)
predicted_labels = [int(round(x[0])) for x in predicted_scores]
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label) # here we goo our model created the unseen comment with right category review


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
in this model we load data
prepare it
train the model
export model
and make prediction with unseen example
'''
