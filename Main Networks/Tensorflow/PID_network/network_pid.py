import tensorflow as tf
import pandas
import numpy as np
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

NAME = "network_pid".format(int(time()))
X = pandas.read_hdf("/home/felix/PycharmProjects/tensorflow_network/particle_data_big.h5")
y = X.pi_TRUEID
X= X.drop("pi_TRUEID", axis = 1)

print(X)
print(y)
y = np.array(y, ndmin=2).T
X = np.array(X, ndmin=2)
for i in range(len(y)):
    if y[i] == (321):
        y[i] = 0
    elif y[i] == (-321):
        y[i] = 0
    elif y[i] == (211):
        y[i] = 1
    elif y[i] == (-211):
        y[i] = 1

test_X = X
X = tf.keras.utils.normalize(
        X,
        axis=-1,
        order=2
        )

print(X)
print(y)

model = keras.Sequential([
    keras.layers.Dense(input_shape=(30,), units=2),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# sets optimization function, learning rate and loss function for network
model.compile(optimizer= tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# trains the network for 1000 epochs with 500 iterations per epoch
model.fit(X,
          y,
          epochs = 1000,
          callbacks=[tensorboard],
          batch_size=100,
          shuffle=True,
          validation_split=0.1
          )


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)




predictions = model.predict(X)
# print(predictions)
k= 0
confusion_matrix = []
for i in range(2):
    confusion_matrix.append([])
    for j in range(2):
        confusion_matrix[i].append(0)


for i in predictions:
   prediction = np.argmax(predictions[k])
   real_result = y[k][0]
   confusion_matrix[prediction][real_result]= confusion_matrix[prediction][real_result]+1
   k = k + 1
#confusion_matrix[0][9] = 'test'
confusion_matrix = np.array(confusion_matrix)
print(confusion_matrix)

