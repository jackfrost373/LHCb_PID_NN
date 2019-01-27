import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
# get big data from the hdf5 file
df = pd.read_hdf('/home/Shared/students/particle_data_big_kaon0_pion1_proton2.h5',stop=100000)

# extract PID class labels
y = df.pi_TRUEID
dfsel = df.drop('pi_TRUEID', axis = 1)
attributesTRACK = ['TrackP','TrackPt','TrackChi2PerDof','TrackLikelihood','TrackFitTChi2','TrackFitTNDoF','TrackFitMatchChi2', 'TrackGhostProbability','TrackCloneDist','TrackFitVeloChi2','TrackFitVeloNDoF',]
attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
attributesDLLS = ['RichDLLe','RichDLLmu','RichDLLk','RichDLLp','RichDLLbt']
attributesCALO = ['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']
attributesOther = ['VeloCharge', 'pi_TRACK_time','pi_TRACK_time_err']
attributes = attributesTRACK + attributesRICH + attributesDLLS + attributesCALO + attributesOther
X = dfsel.loc[ :, attributes ]
print(len(attributes))

# define particle IDs as'labels' for the distinct particle types
y = y.astype('category')

scaler = StandardScaler()
kf=KFold(n_splits=3)
for train_index, test_index in kf.split(X,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
print("Training...")
##Cross Val##
i   = 0
##cross val fin##
#akk=(int(len(attributes)-2), 10)
mlp = MLPClassifier(hidden_layer_sizes=(32,16,8)
                    ,activation='relu', solver='adam',tol=0.001 , max_iter=10,
                    verbose=1., shuffle=True ,learning_rate='constant',learning_rate_init=0.001,n_iter_no_change=10)#_init=stepsize(mlp.loss_curve_),n_iter_no_change=10.)  

for train_indices, test_indices in kf.split(X):
    mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion matrix:")
print(confusion_matrix(y_test,predictions))
print("Classification report:")
print(classification_report(y_test,predictions))
attributes=attributes+[]
print(attributes)    

probabilities = mlp.predict_proba(X_test) # gives list of [prob_pion, prob_kaon]
class_var = np.array( [ probabilities[i][0] for i in range(len(y_test)) ] ) # list of prob_pion
##error beneath##
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

df_test_pions["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="pions", log= "y")
df_test_kaons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="kaons",log= "y")
df_test_protons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="protons", log= "y")
plt.legend(loc='upper center')
plt.xlabel("myMLP classifier")
plt.show()