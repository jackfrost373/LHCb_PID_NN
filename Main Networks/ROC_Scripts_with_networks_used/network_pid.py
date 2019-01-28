import tensorflow as tf
import numpy as np
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
import pandas
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

NAME = "network_pid".format(int(time()))

X = pandas.read_hdf('/home/szymon/PycharmProjects/test/particle_data_big_kaon0_pion1_proton2.h5', stop=100000)
y = X.pi_TRUEID
X = X.drop("pi_TRUEID", axis=1)

print(pandas.DataFrame.keys(X))
print(y.keys)
y = np.array(y, ndmin=2).T
X = np.array(X, ndmin=2)

# for i in range(len(y)):
#     if y[i] == (321):
#         y[i] = 0
#     elif y[i] == (-321):
#         y[i] = 0
#     elif y[i] == (211):
#         y[i] = 1
#     elif y[i] == (-211):
#         y[i] = 1

test_X = X
X = tf.keras.utils.normalize(
        X,
        axis=-1,
        order=2
        )

print(X)
print(y)
#
class_weight = {0: 2,
                1: 16,
                2: 1
                }

# class_weight = class_weight.compute_class_weight('balanced'
#                                                ,np.unique(y)
#                                                ,y)

model = keras.Sequential([
    keras.layers.Dense(input_shape=(32,), units=2),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(64, activation=tf.nn.leaky_relu),
    keras.layers.Dense(32, activation=tf.nn.sigmoid),
    keras.layers.Dense(16, activation=tf.nn.leaky_relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# sets optimization function, learning rate and loss function for network
model.compile(optimizer= tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# trains the network for 1000 epochs with 500 iterations per epoch
model.fit(X,
          y,
          epochs = 200,
          callbacks=[tensorboard],
          batch_size=100,
          shuffle=True,
          validation_split=0.1,
          # class_weight= class_weight,
          verbose=1
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


k= 0
prediction = []
for i in predictions:
   prediction.append(int(np.argmax(predictions[k])))
   real_result = y[k][0]
   k = k + 1
prediction=np.array(prediction, ndmin=2).T

# confusion_matrix = tf.confusion_matrix(y, prediction)
# labels0 = y
# labels1 = y
# labels2 = y
#
# for i in y:
#     if y[i] == 2:
#         labels0[i] = 1
#     elif y[i] == 0:
#         labels1[i] = 2
#     elif y[i] == 1:
#         labels2 = 0

for i in range(3):
    stored_labels = y.T[0]
    temp_labels = []
    temp_predictions = predictions.T[i] #(iterates through the columns)
    for j in range(len(y)):
        if stored_labels[j] == i:
            temp_labels.append(1)
        else:
            temp_labels.append(0)
    fpr_keras0, tpr_keras0, thresholds_keras0 = roc_curve(temp_labels, temp_predictions)
    auc_keras0 = auc(fpr_keras0, tpr_keras0)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras0, tpr_keras0, label='Keras0 (area = {:.3f})'.format(auc_keras0))



plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# with tf.Session():
#     print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))



np.save("np_predictions", predictions)
np.save("np_y", y)