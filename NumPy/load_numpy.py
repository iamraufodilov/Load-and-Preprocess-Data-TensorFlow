# load libraries
import numpy as np
import tensorflow as tf


# load dataset
file_path = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow Core/Load and Preprocess Data TensorFlow/NumPy/mnist.npz'

with np.load(file_path) as data:
    train_examples = data['x_train'],
    train_labels = data['y_train'],
    test_examples = data['x_test'],
    test_labels = data['y_test']


# to create train and test dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))


# shuffle and batch dataset
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
# compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

# train the model
model.fit(train_dataset, epochs=10) #here we go we got 97% accuracy

# evaluate the model
model.evaluate(test_dataset) # and also evaluation also 95% high


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
we load numpy datset
we transform it to tf.data.Dataset
we train the model with high accuracy
'''