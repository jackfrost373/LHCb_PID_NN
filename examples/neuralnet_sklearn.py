
import uproot
import pandas as pd
import matplotlib.pyplot as plt

# Read tree as pandas dataframe using uproot
tfile = uproot.open('../datafiles/output/davinci_MC_PID.root')
tree = tfile["PiTree/DecayTree"]
df = tree.pandas.df()
#print(df.head())


# make cuts on dataset
crit_global = (df['pi_TRACK_time_err'] > 0.1)
crit_types  = (abs(df['pi_TRUEID']) == 211) | (abs(df['pi_TRUEID']) == 2212) # proton or pion
dfsel = df[crit_global & crit_types]

# make new PID label abs
#dfsel['absid'] = dfsel['pi_TRUEID'].abs()
#print(dfsel['absid'].value_counts())



# define 'attributes' to train on:
#attributes = ['VeloCharge','TrackP','TrackPt', 'EcalPIDe','RichDLLp','pi_TRACK_time']
#X = dfsel.loc[ :, attributes ]

# define 'labels'
#y = dfsel.loc[ :, 'pi_TRUEID' ]
#print(y.pi_TRUEID.unique())



