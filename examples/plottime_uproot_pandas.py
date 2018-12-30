
import uproot
import pandas as pd
import matplotlib.pyplot as plt

# Read tree as pandas dataframe using uproot
tfile = uproot.open('../datafiles/output/davinci_MC_PID.root')
tree = tfile["PiTree/DecayTree"]
df = tree.pandas.df()

# make cuts on dataset
crit_global = (df['pi_P'] < 7000) & (df['pi_TRACK_time_err'] > 0.1)
crit_pion   = abs(df['pi_TRUEID']) == 211
crit_proton = abs(df['pi_TRUEID']) == 2212
df_pions   = df[crit_global & crit_pion]
df_protons = df[crit_global & crit_proton]

# make plots using matplotlib
df_pions[  "pi_TRACK_time"].plot.hist(bins=100, range=(-3,3), alpha=0.5, density=True, label="pions")
df_protons["pi_TRACK_time"].plot.hist(bins=100, range=(-3,3), alpha=0.5, density=True, label="protons")
plt.legend(loc='upper right')
plt.xlabel("track time [ns]")
plt.show()

f = plt.figure()
f.savefig("track_time_uproot-pandas.pdf", bbox_inches='tight')
