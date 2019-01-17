# Necessary libraries
import uproot
import pandas as pd

file = uproot.open('/home/Shared/lhcbdata/ganga/20/1/output/davinci_MC_PID.root') # Opens a .root file with simulated data.

tree = file["PiTree/DecayTree"] # Takes the tree from this .root file.
df = tree.pandas.df() # Turns the tree data into a pandas dataframe.

tracking = df[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']] # Desired tracking variables.
RICH = df[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']] # Desired RICH variables.
CALO = df[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']] # Desired CALO variable.
VELO = df[['VeloCharge']] # Desired VELO variable.
ID = df[['pi_TRUEID']] # Particle IDs, known from simulation.
data = pd.concat([tracking, RICH, CALO, VELO, ID], axis = 1) # Strings all variables and the particle IDs together into one dataframe.

# Filtering trick from https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/
data_kaon_pion = data[(data.pi_TRUEID == 211) | (data.pi_TRUEID == -211) | (data.pi_TRUEID == 321) | (data.pi_TRUEID == -321)]

data_kaon_pion_hdf5 = data_kaon_pion.to_hdf('particle_data_big.h5', key = 'kaon_pion', format = 'table')