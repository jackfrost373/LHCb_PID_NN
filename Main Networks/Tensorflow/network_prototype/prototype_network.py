import numpy as np
import pandas
import tensorflow as tf
from tensorflow import keras

# read in dataset
X = pandas.read_csv("/home/felix/PycharmProjects/tensorflow_network/winequality-white.csv",
                           header=0,
                           delimiter=';',
                           usecols=[0,1,2,3,4,5,6,7,8,9,10])
# convert to numpy array
X = np.array(X)

# read in 'solutions' (value that's supposed to be predicted)
y = pandas.read_csv("/home/felix/PycharmProjects/tensorflow_network/winequality-white.csv",
                           header=0,
                           delimiter=';',
                            usecols=[11])
# convert to numpy array
y = np.array(y)


# print dataset for analytics
print(X)
print(y)

# creates model of neural network

model = keras.Sequential([
    keras.layers.Dense(11, activation=tf.nn.relu),
    keras.layers.Dense(12, activation=tf.nn.relu),
    keras.layers.Dense(11, activation=tf.nn.sigmoid)
])

# sets optimization function, learning rate and loss function for network
model.compile(optimizer= tf.train.AdamOptimizer(learning_rate=0.01),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# trains the network for 1000 epochs with 500 iterations per epoch
model.fit(X,
          y,
          epochs = 1000,
          batch_size=500,
          )

