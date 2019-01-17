import tensorflow as tf
import pandas
import numpy as np
from tensorflow import keras

CATEBORIES = ["Pion", "Kaon"]
X_pion = pandas.read_hdf("/home/felix/PycharmProjects/tensorflow_network/particle_data _prelim.h5", 'pion')
X_kaon = pandas.read_hdf("/home/felix/PycharmProjects/tensorflow_network/particle_data _prelim.h5", 'kaon')
X_pion = np.array(X_pion)
X_kaon = np.array(X_kaon)
y=[]
for i in X_kaon:
   y.append(0)
for i in X_pion:
   y.append(1)
y = np.array(y, ndmin=2).T

X = np.concatenate((X_pion, X_kaon))


print(X)
print(y)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])

# sets optimization function, learning rate and loss function for network
model.compile(optimizer= tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# trains the network for 1000 epochs with 500 iterations per epoch
model.fit(X,
          y,
          epochs = 100,
          batch_size=1000,
          shuffle=True,
          )