import tensorflow as tf
import numpy as np
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
import pandas


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

NAME = "network_pid".format(int(time()))
# infile = open('Dataset_without_zeroes','rb')
# X = pickle.load(infile)
# infile.close()
# infile = open('labels_without_zeroes','rb')
# y = pickle.load(infile)
# infile.close()

X = pandas.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_chungus_kaon0_pion1.h5')
y = X.pi_TRUEID
X = X.drop("pi_TRUEID", axis=1)

print(X)
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

model = keras.Sequential([
    keras.layers.Dense(input_shape=(30,), units=2),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    keras.layers.Dense(64, activation=tf.nn.leaky_relu),
    keras.layers.Dense(32, activation=tf.nn.leaky_relu),
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
          epochs = 10,
          callbacks=[tensorboard],
          batch_size=10000,
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
roc_predictions = predictions.T
roc_predictions = roc_predictions[1]
#roc curve


fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, roc_predictions)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()
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
#np.savetxt('predictions', predictions)
#np.savetxt('labels', y)
print(confusion_matrix)
perc_kaon = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])*100
perc_pion = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])*100
print("Percentage of correctly identified kaons:", perc_kaon)
print("Percentage of correctly identified pions:", perc_pion)

