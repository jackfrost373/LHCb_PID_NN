import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get big data from the hdf5 file
df = pd.read_hdf('/home/felix/PycharmProjects/tensorflow_network/particle_data_chungus_kaon0_pion1.h5', stop=100000)

# extract PID class labels
y = df.pi_TRUEID

# make some cuts on the data
#crit_global = ( df['pi_P'] > 1200) #insert threshold values for track time error and momentum
#dfsel = df[crit_global]

# remove true_ID column from data before entering training
dfsel = df.drop('pi_TRUEID', axis = 1)

# define 'attributes' to train on: (i.e. which variables?)
attributesTRACK = ['TrackP','TrackPt','TrackChi2PerDof','TrackLikelihood','TrackFitTChi2','TrackFitTNDoF','TrackFitMatchChi2',
                     'TrackGhostProbability','TrackCloneDist','TrackFitVeloChi2','TrackFitVeloNDoF',]
attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
attributesDLLS = ['RichDLLe','RichDLLmu','RichDLLk','RichDLLp','RichDLLbt']
attributesCALO = ['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']
attributesOther = ['VeloCharge']#, 'pi_TRACK_time','pi_TRACK_time_err']
attributes = attributesTRACK + attributesRICH + attributesDLLS + attributesCALO + attributesOther
X = dfsel.loc[ :, attributes ]

# define particle IDs as'labels' for the distinct particle types
y = y.astype('category')

# split sample into training and testing sets w/ an inbuilt scikit function
#for that: input data and the labels that classify it
from sklearn.model_selection import KFold
kf=KFold(n_splits=2)         #select no of splits
for train_index, test_index in kf.split(X):                              #this is used instead of standard scaler
 #   print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]#.transform(X_test)

#train with My little pony...
print("Training...")
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(len(attributes), int(len(attributes)/2), 10), max_iter=200, activation='relu', solver='adam', verbose=1)
for train_indices, test_indices in kf.split(X):                    #no of splits on dataset
    mlp.fit(X_test,y_test)

    print(mlp.score(X_test,y_test))

# make label predictions on test data
predictions = mlp.predict(X_test)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion matrix:")
print(confusion_matrix(y_test,predictions))
print("Classification report:")
print(classification_report(y_test,predictions))

# create a classification variable based on the prediction probability
probabilities = mlp.predict_proba(X_test) # gives list of [prob_pion, prob_kaon]
#class_var = np.array( [ probabilities[i][0] for i in range(len(y_test)) ] ) # list of prob_pion
y_labels = np.array([y_test[:][i] for i in range(len(y_test)) ] ) #this is the original testing data
# y_labels_b = (y_labels==211).astype(int)
#
# df_test = pd.DataFrame(data=X_test, columns=attributes)
# df_test["myMLP"] = class_var
# df_test["absid"] = y_test
#
# # plot this variable for the 3 categories, aka particle types
# crit_pion = df_test['absid'] == 0
# crit_kaon = df_test['absid'] == 1
# #crit_proton = df_test['absid'] == 2
# df_test_pions = df_test[crit_pion]
# df_test_kaons = df_test[crit_kaon]
# #df_test_protons = df_test[crit_proton]
#
# df_test_pions["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="pions")
# df_test_kaons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="kaons")
# #df_test_protons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="protons")
# plt.legend(loc='upper right')
# plt.xlabel("myMLP classifier")
# plt.show()
#
# f = plt.figure()
# f.savefig("myMLP_performance.pdf", bbox_inches='tight')

np.save("scikit_predictions", probabilities)
np.save("scikit_y", y_labels)