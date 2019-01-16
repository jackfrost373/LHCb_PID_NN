import numpy as np
import random
import pandas
import tensorflow as tf
from tensorflow import keras

X = pandas.read_csv("/home/felix/PycharmProjects/tensorflow_network/winequality-white.csv",
                           header=0,
                           delimiter=';',
                           usecols=[0,1,2,3,4,5,6,7,8,9,10])
X = np.array(X)
y = pandas.read_csv("/home/felix/PycharmProjects/tensorflow_network/winequality-white.csv",
                           header=0,
                           delimiter=';',
                            usecols=[11])
y = np.array(y)



print(X)
print(y)

model = keras.Sequential([
    keras.layers.Dense(11, activation=tf.nn.relu),
    keras.layers.Dense(12, activation=tf.nn.relu),
    keras.layers.Dense(11, activation=tf.nn.sigmoid)
])

model.compile(optimizer= tf.train.AdamOptimizer(learning_rate=0.01),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X,
          y,
          epochs = 1000,
          batch_size=500,
          )

