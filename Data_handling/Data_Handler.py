# Import all necessary code libraries
import uproot # To read .root files and transform them into pandas dataframes.
import pandas as pd # To list and select the data we want for the neural network.
import sys # To be able to provide input arguments.

filename = sys.argv[1] # Collects the name/path of the file to extract data from when this is transferred to a python script.

file = uproot.open(filename) # Open desired data file. Could make this variable input.

# Extract neural network data for kaons.
kaon_tree = file["DzTree_Kaon/DecayTree"] # Select data to pick variables from.
kaon_dataframe = kaon_tree.pandas.df() # Turns all data into a dataframe that pandas can work with.
# Extract the  kaon variables used for the neural net. Detector by detector first, then concatenate.
kaon_tracking = kaon_dataframe[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']]
kaon_RICH = kaon_dataframe[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']]
kaon_CALO = kaon_dataframe[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']]
kaon_VELO = kaon_dataframe[['VeloCharge']]
#kaon_data = pd.concat([kaon_tracking, kaon_RICH, kaon_CALO, kaon_VELO], axis = 1)
kaon_hdf5 = kaon_data.to_hdf('kaon_data.h5', key = 'kaon', format = 'table')

# Extract neural network data for pions.
pion_tree = file["DzTree_Pion/DecayTree"] # Select data to pick variables from.
pion_dataframe = pion_tree.pandas.df() # Turns all data into a dataframe that pandas can work with.
# Extract the  pion variables used for the neural net. Detector by detector first, then concatenate.
pion_tracking = pion_dataframe[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']]
pion_RICH = pion_dataframe[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']]
pion_CALO = pion_dataframe[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']]
pion_VELO = pion_dataframe[['VeloCharge']]
pion_data = pd.concat([pion_tracking, pion_RICH, pion_CALO, pion_VELO], axis = 1)
#pion_data

# Extract neural network data for protons.
proton_tree = file["LcTree_Proton/DecayTree"] # Select data to pick variables from.
proton_dataframe = proton_tree.pandas.df() # Turns all data into a dataframe that pandas can work with.
# Extract the proton variables used for the neural net. Detector by detector first, then concatenate.
proton_tracking = proton_dataframe[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']]
proton_RICH = proton_dataframe[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']]
proton_CALO = proton_dataframe[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']]
proton_VELO = proton_dataframe[['VeloCharge']]
proton_data = pd.concat([proton_tracking, proton_RICH, proton_CALO, proton_VELO], axis = 1)
#proton_data