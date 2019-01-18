import tensorflow as tf
import pandas
import numpy as np
from tensorflow import keras

CATEBORIES = ["Pion", "Kaon"]
X = pandas.read_hdf("/home/felix/PycharmProjects/tensorflow_network/particle_data_big.h5")

y = 0

y = X.pi_TRUEID
print(X)
print(y)
y = np.array(y, ndmin=2).T
X = np.array(X, ndmin=2)
for i in range(len(y)):
    if y[i] == (321):
        y[i] = 0
    elif y[i] == (-321):
        y[i] = 1
    elif y[i] == (211):
        y[i] = 2
    elif y[i] == (-211):
        y[i] = 3




print(X)
print(y)

model = keras.Sequential([
    keras.layers.Dense(input_shape=(31,), units=2),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.sigmoid)
])

# sets optimization function, learning rate and loss function for network
model.compile(optimizer= tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# trains the network for 1000 epochs with 500 iterations per epoch
model.fit(X,
          y,
          epochs = 100,
          batch_size=10000,
          shuffle=True,
          )