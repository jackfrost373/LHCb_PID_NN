import matplotlib.pyplot as plt
import pandas
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd

X = pandas.read_hdf('/home/felix/PycharmProjects/FinalPID/particle_data_big_kaon0_pion1_proton2.h5', stop=100000)
y = X.pi_TRUEID

target_results_tensorflow =  np.load("np_y.npy")
roc_data_tensorflow =np.load("np_predictions.npy")
target_results_sklearn = np.load("np_y_SKlearn.npy")
roc_data_sklearn = np.load("np_prediction_SKlearn.npy")

attributesTRACK = ['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackFitTChi2', 'TrackFitTNDoF',
                   'TrackFitMatchChi2', 'TrackGhostProbability', 'TrackCloneDist', 'TrackFitVeloChi2',
                   'TrackFitVeloNDoF', ]
attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
attributesDLLS = ['RichDLLe', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']
attributesCALO = ['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']
attributesOther = ['VeloCharge', 'pi_TRACK_time', 'pi_TRACK_time_err']
attributes = attributesTRACK + attributesRICH + attributesDLLS + attributesCALO + attributesOther

# create a classification variable based on the prediction probability
class_var = np.array( [roc_data_tensorflow[i][0] for i in range(len(y)) ] ) # list of prob_pion
df_test = pd.DataFrame(data=X, columns=attributes)
df_test["myMLP"] = class_var
df_test["absid"] = y

# plot this variable for the two categories
crit_pion = df_test['absid'] == 1
crit_kaon = df_test['absid'] == 0
crit_proton = df_test['absid'] == 2
df_test_pions = df_test[crit_pion]
df_test_kaons = df_test[crit_kaon]
df_test_protons = df_test[crit_proton]

df_test_pions["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="pions", log="y")
df_test_kaons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="kaons", log="y")
df_test_protons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="protons")
plt.legend(loc='upper right')
plt.xlabel("myMLP classifier")
plt.show()

f = plt.figure()
f.savefig("myMLP_performance.pdf", bbox_inches='tight')

# create a classification variable based on the prediction probability
class_var = np.array( [roc_data_sklearn[i][0] for i in range(len(target_results_sklearn)) ] ) # list of prob_pion
df_test = pd.DataFrame(data=X, columns=attributes)
df_test["myMLP"] = class_var
df_test["absid"] = target_results_sklearn

# plot this variable for the two categories
crit_pion = df_test['absid'] == 1
crit_kaon = df_test['absid'] == 0
crit_proton = df_test['absid'] == 2
df_test_pions = df_test[crit_pion]
df_test_kaons = df_test[crit_kaon]
df_test_protons = df_test[crit_proton]

df_test_pions["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="pions", log="y")
df_test_kaons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="kaons", log="y")
df_test_protons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="protons")
plt.legend(loc='upper right')
plt.xlabel("myMLP classifier")
plt.show()

f = plt.figure()
f.savefig("myMLP_performance.pdf", bbox_inches='tight')