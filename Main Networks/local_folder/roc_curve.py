import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import pandas

roc_predictions = np.load("/home/felix/PycharmProjects/tensorflow_network/venv/scikit_predictions.npy")
labels = np.load("/home/felix/PycharmProjects/tensorflow_network/venv/scikit_y.npy")
target_results = np.load("/home/felix/PycharmProjects/tensorflow_network/venv/np_y.npy")
roc_data = np.load("/home/felix/PycharmProjects/tensorflow_network/venv/np_predictions.npy")
probNN_labels = pandas.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_ProbNN_big_kaon0_pion1_proton2.h5',
                                columns= ['pi_TRUEID'],
                                stop=100000,
                                )
mc12_predictions = pandas.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_ProbNN_big_kaon0_pion1_proton2.h5',
                                    columns= ['pi_MC12TuneV4_ProbNNk', 'pi_MC12TuneV4_ProbNNp'],
                                    stop=100000,
                                    )
mc15_predictions = pandas.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_ProbNN_big_kaon0_pion1_proton2.h5',
                                    columns= ['pi_MC15TuneV1_ProbNNk', 'pi_MC15TuneV1_ProbNNp'],
                                    stop=100000,
                                    )
DNNV_predictions = pandas.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_ProbNN_big_kaon0_pion1_proton2.h5',
                                    columns= ['pi_MC15TuneDNNV1_ProbNNk', 'MC15TuneDNNV1_ProbNNp'],
                                    stop=100000,
                                    )
FLAT_predictions = pandas.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_ProbNN_big_kaon0_pion1_proton2.h5',
                                    columns= ['pi_MC15TuneFLAT4dV1_ProbNNk', 'MC15TuneFLAT4dV1_ProbNNp'],
                                    stop=100000,
                                    )
CATBOOST_predictions = pandas.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_ProbNN_big_kaon0_pion1_proton2.h5',
                                    columns= ['pi_MC15TuneCatBoostV1_ProbNNk', 'pi_MC15TuneCatBoostV1_ProbNNp'],
                                    stop=100000,
                                    )
# MC15TuneDNNV1, MC15TuneFLAT4dV1, and MC15TuneCatBoostV1
roc_predictions = roc_predictions.T[1]
mc15_predictions = np.array(mc15_predictions)
mc15_predictions = mc15_predictions.T[0]
mc12_predictions = np.array(mc12_predictions)
mc12_predictions = mc12_predictions.T[0]
DNNV_predictions = np.array(DNNV_predictions)
DNNV_predictions = mc12_predictions.T
FLAT_predictions = np.array(FLAT_predictions)
FLAT_predictions = mc12_predictions.T
CATBOOST_predictions = np.array(CATBOOST_predictions)
CATBOOST_predictions = mc12_predictions.T
probNN_labels = np.array(probNN_labels)
print("step 1")
for x in reversed(range(len(probNN_labels))):
    if probNN_labels[x] == 2:
        probNN_labels = np.delete(probNN_labels, x, 0)
        mc12_predictions = np.delete(mc12_predictions, x, 0)
        mc15_predictions = np.delete(mc15_predictions, x, 0)
        DNNV_predictions = np.delete(DNNV_predictions, x, 0)
        FLAT_predictions = np.delete(FLAT_predictions, x, 0)
        CATBOOST_predictions = np.delete(CATBOOST_predictions, x, 0)
        print(x)
print("step 2")
roc_data = np.array(roc_data).T
roc_data = roc_data[0]
labels = np.array(labels)


fpr_sklearn, tpr_sklearn, thresholds_sklearn = roc_curve(labels, roc_predictions)
tpr_keras, fpr_keras, thresholds_keras = roc_curve(target_results, roc_data)
tpr_12, fpr_12, thresholds_12 = roc_curve(probNN_labels, mc12_predictions)
tpr_15, fpr_15, thresholds_15 = roc_curve(probNN_labels, mc15_predictions)
tpr_DNNV, fpr_DNNV, thresholds_DNNV = roc_curve(probNN_labels, DNNV_predictions)
tpr_FLAT, fpr_FLAT, thresholds_FLAT = roc_curve(probNN_labels, FLAT_predictions)
tpr_CATBOOST, fpr_CATBOOST, thresholds_CATBOOST = roc_curve(probNN_labels, CATBOOST_predictions)


auc_keras = auc(fpr_keras, tpr_keras)
auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
auc_12 = auc(fpr_12, tpr_12)
auc_15 = auc(fpr_15,tpr_15)
auc_DNNV = auc(fpr_DNNV,tpr_DNNV)
auc_CATBOOST = auc(fpr_CATBOOST,tpr_CATBOOST)
auc_FLAT = auc(fpr_FLAT,tpr_FLAT)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_sklearn, tpr_sklearn, label ='sklearn (area = {:.3f})'.format(auc_sklearn))
plt.plot(fpr_12, tpr_12, label ='MC12 (area = {:.3f})'.format(auc_12))
plt.plot(fpr_15, tpr_15, label ='MC15 (area = {:.3f})'.format(auc_15))
plt.plot(fpr_DNNV, tpr_DNNV, label ='DNNV (area = {:.3f})'.format(auc_DNNV))
plt.plot(fpr_FLAT, tpr_FLAT, label ='FLAT (area = {:.3f})'.format(auc_FLAT))
plt.plot(fpr_CATBOOST, tpr_CATBOOST, label ='CATBOOST (area = {:.3f})'.format(auc_CATBOOST))


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()