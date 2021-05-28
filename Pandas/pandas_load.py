# load libraries
import pandas as pd
import tensorflow as tf


#load dtaset
df_path = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow Core/Load and Preprocess Data TensorFlow/Pandas/heart.csv'
df = pd.read_csv(df_path)
#_>print(df.head())
#_>print(df.dtypes) # here we can see only thal collumn does not has stndart type so lets change it


# change thal collumn type
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes


# get label and feature
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ)) # here you can see our featue and target labels


tf.constant(df['thal'])


# shuffle and batch dataset
train_dataset = dataset.shuffle(len(df)).batch(1)


# create model function
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model


#train the model
model = get_compiled_model()
#_>model.fit(train_dataset, epochs=15)


# Alternative to feature columns
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
  print (dict_slice) # here you can see we get dictionary which key is collumn name value is row values


# train the model
model_func.fit(dict_slices, epochs=15)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
we load dataset as pd dataframe
we train the data
in second part 
we change data structure to dctionry structure where collumn key and value is feature 
'''