# This script opens NellieW's big datafile and trains a SciKit learn neural network on it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get big data from the hdf5 file
df = pd.read_hdf('/home/Shared/students/particle_data_big_kaon0_pion1_proton2.h5')

# extract PID class labels
y = df.pi_TRUEID

# make some cuts on the data
crit_global = ( df['pi_P'] > 1200) #insert threshold values for track time error and momentum
dfsel = df[crit_global]

# remove true_ID column from data before entering training
dfsel = df.drop('pi_TRUEID', axis = 1)

# define 'attributes' to train on: (i.e. which variables?)
attributesTRACK = ['TrackP','TrackPt','TrackChi2PerDof','TrackNumDof','TrackLikelihood','TrackFitTChi2','TrackFitTNDoF','TrackFitMatchChi2',
                     'TrackGhostProbability','TrackCloneDist','TrackFitVeloChi2','TrackFitVeloNDoF',]
attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
attributesDLLS = ['RichDLLe','RichDLLmu','RichDLLk','RichDLLp','RichDLLbt']
attributesCALO = ['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']
attributesOther = ['VeloCharge', 'pi_TRACK_time','pi_TRACK_time_err']
attributes = attributesTRACK + attributesRICH + attributesDLLS + attributesCALO + attributesOther
X = dfsel.loc[ :, attributes ]

# define particle IDs as'labels' for the distinct particle types
y = y.astype('category')

# split sample into training and testing sets w/ an inbuilt scikit function
#for that: input data and the labels that classify it
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

########################### Now follows the machine learning bit...
# feature scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#train with My little pony... 
print("Training...") 
from sklearn.neural_network import MLPClassifier 
mlp = MLPClassifier(hidden_layer_sizes=(len(attributes), int(len(attributes)/2), 10), max_iter=100, activation='relu', solver='adam', verbose=1)
mlp.fit(X_train, y_train.values.ravel())

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
class_var = np.array( [ probabilities[i][0] for i in range(len(y_test)) ] ) # list of prob_pion
y_labels = np.array([y_test[:][i] for i in range(len(y_test)) ] ) #this is the original testing data 
y_labels_b = (y_labels==211).astype(int)

df_test = pd.DataFrame(data=X_test, columns=attributes)
df_test["myMLP"] = class_var
df_test["absid"] = y_test

# plot this variable for the 3 categories, aka particle types
crit_pion = df_test['absid'] == 211
crit_kaon = df_test['absid'] == 321
crit_proton = df_test['absid'] == 2212
df_test_pions = df_test[crit_pion]
df_test_kaons = df_test[crit_kaon]
df_test_protons = df_test[crit_proton]

df_test_pions["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="pions")
df_test_kaons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="kaons")
df_test_protons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="protons")
plt.legend(loc='upper right')
plt.xlabel("myMLP classifier")
plt.show()

f = plt.figure()
f.savefig("myMLP_performance.pdf", bbox_inches='tight')

