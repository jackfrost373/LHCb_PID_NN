import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import pandas


target_results_tensorflow =  np.load("np_y.npy")
roc_data_tensorflow =np.load("np_predictions.npy")
target_results_sklearn = np.load("np_y_SKlearn.npy")
roc_data_sklearn = np.load("np_prediction_SKlearn.npy")

for i in range(3):
    stored_labels_tensorflow = target_results_tensorflow.T[0]
    temp_labels_tensorflow = []
    temp_predictions_tensorflow = roc_data_tensorflow.T[i] #(iterates through the columns)

    stored_labels_sklearn = target_results_sklearn.T[0]
    temp_labels_sklearn = []
    temp_predictions_sklearn = roc_data_sklearn.T[i]  # (iterates through the columns)
    for j in range(len(target_results_tensorflow)):
        if stored_labels_tensorflow[j] == i:
            temp_labels_tensorflow.append(1)
        else:
            temp_labels_tensorflow.append(0)
    for k in range(len(target_results_sklearn)):
        if stored_labels_sklearn[k] == i:
            temp_labels_sklearn.append(1)
        else:
            temp_labels_sklearn.append(0)
    # if i == 2:
    #     tpr_sklearn, fpr_sklearn, thresholds_sklearn = roc_curve(temp_labels_sklearn, temp_predictions_sklearn)
    # else:
    fpr_sklearn, tpr_sklearn, thresholds_sklearn = roc_curve(temp_labels_sklearn, temp_predictions_sklearn)
    auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(temp_labels_tensorflow, temp_predictions_tensorflow)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_sklearn, tpr_sklearn, label='SKlearn (area = {:.3f})'.format(auc_sklearn))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()