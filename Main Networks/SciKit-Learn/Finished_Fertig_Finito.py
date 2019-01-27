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
print(df.head())

# extract PID class labels
y = df.pi_TRUEID
df['absid'] = df['pi_TRUEID'].abs()
print(df['absid'].value_counts())
dfsel = df.drop('pi_TRUEID',axis = 1)
dfsel = df.drop('RichDLLmu', axis=1)
attributesTRACK = ['TrackP','TrackPt','TrackChi2PerDof','TrackLikelihood','TrackFitTChi2','TrackFitTNDoF','TrackFitMatchChi2', 'TrackGhostProbability','TrackCloneDist','TrackFitVeloChi2','TrackFitVeloNDoF',]
attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
attributesDLLS = ['RichDLLe','RichDLLk','RichDLLp','RichDLLbt']
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
dftst_Pions= [0,0,0,0,0,0,]##i know not the best way to initialize arrays of unknown size
dftst_Kaons=[0,0,0,0,0,0,]
gl  = [0,0,0,0,0,0,]
prob= [0,0,0,0,0,0,]
pred= [0,0,0,0,0,0,]
c_v = [0,0,0,0,0,0,]
dftst=[0,0,0,0,0,0,]

def cross(i):
    
    g=mlp.score(X_test, y_test)
    gl[i]=g
    prob[i] = mlp.predict_proba(X_test)
    pred[i] = mlp.predict(X_test)
    c_v[i]  = np.array( [ prob[i][c][0] for c in range(len(y_test)) ] )
    dftst[i] =  pd.DataFrame(data=X_test, columns=attributes)
    dftst[i]["MyMLP"]=c_v[i]
    dftst[i]["absid"]=y_test
    print("Training...")
##Cross Val##
i   = 0
##cross val fin##
#akk=(int(len(attributes)-2), 10)
mlp = MLPClassifier(hidden_layer_sizes=(128,64,32,16)
                    ,activation='relu', solver='adam',tol=0.001 , max_iter=500,
                    verbose=1., shuffle=True ,learning_rate='constant',learning_rate_init=0.002,n_iter_no_change=30)#_init=stepsize(mlp.loss_curve_),n_iter_no_change=10.)  
#mlp.fit(X_train, y_train.values.ravel()) 
#
for train_indices, test_indices in kf.split(X):
    mlp.fit(X_train, y_train.values.ravel())
    cross(i)
    i=i+1
i=0 
string1="pions"
string2="kaons"
for train_indices, test_indices in kf.split(X):
    
    critPion = dftst[i]["absid"] == 0#amount of data splits with a max of 4
    critKaon = dftst[i]["absid"] == 1
    dftst_Pions[i] = dftst[i][critPion]
    dftst_Kaons[i] = dftst[i][critKaon]
    dftst_Pions[i]["MyMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label=string1+str(i), log="y")
    dftst_Kaons[i]["MyMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label=string2+str(i), log="y")
    i=i+1
plt.legend(loc='upper center')
plt.xlabel("myMLP classifier")
plt.show()
predictions = mlp.predict(X_test)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion matrix:")
print(confusion_matrix(y_test,predictions))
print("Classification report:")
print(classification_report(y_test,predictions))
prob = mlp.predict_proba(X_test)
c_v  = np.array( [ prob[c][0] for c in range(len(y_test)) ] )
dftst =  pd.DataFrame(data=X_test, columns=attributes)
dftst["MyMLP"]=c_v
dftst["absid"]=y_test
critPion = dftst["absid"] == 0#amount of data splits with a max of 4
critKaon = dftst["absid"] == 1
crit_Proton = dftst["absid"] == 2
dftst_Pions = dftst[critPion]
dftst_Kaons = dftst[critKaon]
dftst_Protons = dftst[crit_Proton]
dftst_Pions["MyMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="pions", log="n")
dftst_Kaons["MyMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="kaons", log="n")
dftst_Protons["MyMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="protons", log= "n")
plt.legend(loc='upper center')
plt.xlabel("myMLP classifier")
plt.show()
y_labels = np.array([y_test[:][i] for i in range(len(y_test)) ] ) #this is the original testing data 
y_labels_b = (y_labels==0).astype(int)
fpr,tpr,thresholds = roc_curve(y_labels_b,c_v)
AUCC=auc(fpr,tpr)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='SKlearn (SKlearn = {:.3f})'.format(AUCC))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()