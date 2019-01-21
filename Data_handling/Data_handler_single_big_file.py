# Necessary libraries
import uproot
import pandas as pd

print("running")
file = uproot.open('/home/Shared/lhcbdata/ganga/20/1/output/davinci_MC_PID.root') # Opens a .root file with simulated data.

tree = file["PiTree/DecayTree"] # Takes the tree from this .root file.
df = tree.pandas.df() # Turns the tree data into a pandas dataframe.

tracking = df[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']] # Desired tracking variables.
RICH = df[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']] # Desired RICH variables.
CALO = df[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']] # Desired CALO variable.
VELO = df[['VeloCharge']] # Desired VELO variable.
ID = df[['pi_TRUEID']] # Particle IDs, known from simulation.
data = pd.concat([tracking, RICH, CALO, VELO, ID], axis = 1) # Strings all variables and the particle IDs together into one dataframe.

data_kaon_pion = data[(data.pi_TRUEID == -211) | (data.pi_TRUEID == 211) | (data.pi_TRUEID == -321) | (data.pi_TRUEID == 321)]
print(data_kaon_pion.shape)
# replace each value by what it should be
# TAKE CARE: OTHER PARTICLES NOT REMOVED YET, SO SOME PARTICLES WITH ACTUAL ID 1/0 MIGHT BE HIDING
data_kaon_pion = data_kaon_pion.replace(to_replace= 211, value= 1) # Pions get the ID 1.
data_kaon_pion = data_kaon_pion.replace(to_replace= -211, value= 1) # Antipions get ID 1.
data_kaon_pion = data_kaon_pion.replace(to_replace= 321, value= 0) # Kaons get ID 0.
data_kaon_pion = data_kaon_pion.replace(to_replace= -321, value= 0) # Antikaons get ID 0.
print(data_kaon_pion[['pi_TRUEID']])

data_kaon_pion_hdf5 = data_kaon_pion.to_hdf('particle_data_big_kaon0_pion1.h5', key = 'kaon_pion', format = 'table')
