
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read tree as pandas dataframe using uproot
tfile = uproot.open('../datafiles/output/davinci_MC_PID.root')
tree = tfile["PiTree/DecayTree"]
df = tree.pandas.df()
#print(df.head())

# make new PID label abs
df['absid'] = df['pi_TRUEID'].abs()
#print(df['absid'].value_counts())

# make cuts on dataset
crit_global = (df['pi_TRACK_time_err'] > 0.1) & (df['pi_P'] > 1200)
crit_types  = (abs(df['pi_TRUEID']) == 211) | (abs(df['pi_TRUEID']) == 321) # only kaon or pion
dfsel = df[crit_global & crit_types]
# pion = 211, kaon = 321, proton = 2212


# define 'attributes' to train on:
attributesTRACK = ['TrackP','TrackPt','TrackChi2PerDof','TrackNumDof','TrackLikelihood']
attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
attributesDLLS = ['RichDLLe','RichDLLmu','RichDLLk','RichDLLp','RichDLLbt']
#attributesMuon = ['MuonBkgLL', 'MuonMuLL', 'MuonIsMuon', 'MuonNShared', 'InAccMuon', 'MuonIsLooseMuon']
attributesCALO = ['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']
attributesOther = ['VeloCharge', 'pi_TRACK_time','pi_TRACK_time_err']
attributes = attributesTRACK + attributesRICH + attributesDLLS + attributesCALO + attributesOther
X = dfsel.loc[ :, attributes ]
#print(X.head())

# define 'labels'
y = dfsel['absid'].astype('category')
#print(y.head())


# split sample into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# feature scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# train classifier on train data
print("Training...")
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(len(attributes), int(len(attributes)/2), 10), max_iter=200, activation='relu', solver='adam', verbose=1)  
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
df_test = pd.DataFrame(data=X_test, columns=attributes)
df_test["myMLP"] = class_var
df_test["absid"] = y_test

# plot this variable for the two categories
crit_pion = df_test['absid'] == 211
crit_kaon = df_test['absid'] == 321
df_test_pions = df_test[crit_pion]
df_test_kaons = df_test[crit_kaon]

df_test_pions["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="pions")
df_test_kaons["myMLP"].plot.hist(bins=50, range=(0,1), alpha=0.5, density=True, label="kaons")
plt.legend(loc='upper right')
plt.xlabel("myMLP classifier")
plt.show()

f = plt.figure()
f.savefig("myMLP_performance.pdf", bbox_inches='tight')



