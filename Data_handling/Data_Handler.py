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
kaon_data = pd.concat([kaon_tracking, kaon_RICH, kaon_CALO, kaon_VELO], axis = 1)
kaon_hdf5 = kaon_data.to_hdf('particle_data.h5', key = 'kaon', format = 'table')

# Extract neural network data for pions.
pion_tree = file["DzTree_Pion/DecayTree"] # Select data to pick variables from.
pion_dataframe = pion_tree.pandas.df() # Turns all data into a dataframe that pandas can work with.
# Extract the  pion variables used for the neural net. Detector by detector first, then concatenate.
pion_tracking = pion_dataframe[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']]
pion_RICH = pion_dataframe[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']]
pion_CALO = pion_dataframe[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']]
pion_VELO = pion_dataframe[['VeloCharge']]
pion_data = pd.concat([pion_tracking, pion_RICH, pion_CALO, pion_VELO], axis = 1)
pion_hdf5 = pion_data.to_hdf('particle_data.h5', key = 'pion', format = 'table')

# Extract neural network data for protons.
proton_tree = file["LcTree_Proton/DecayTree"] # Select data to pick variables from.
proton_dataframe = proton_tree.pandas.df() # Turns all data into a dataframe that pandas can work with.
# Extract the proton variables used for the neural net. Detector by detector first, then concatenate.
proton_tracking = proton_dataframe[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']]
proton_RICH = proton_dataframe[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']]
proton_CALO = proton_dataframe[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']]
proton_VELO = proton_dataframe[['VeloCharge']]
proton_data = pd.concat([proton_tracking, proton_RICH, proton_CALO, proton_VELO], axis = 1)
proton_data

#folder = sys.argv[1]

#def NNdata (num): # Opens a .root file, makes it a dataframe and extracts the values we want for the neural network.
#    file = uproot.open('/home/Shared/lhcbdata/ganga/20/'+num+'/output/davinci_MC_PID.root') # Open file.
#    tree = file["PiTree/DecayTree"] #Access decay tree.
#    df = tree.pandas.df() # Turn the ROOTDirectory into a dataframe.
#    tracking = df[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF']] # Wanted variables from the tracker.
#    RICH = df[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']] # Wanted data from the RICH detector.
#    CALO = df[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']] # Wanted data from the CALO detector.
#    VELO = df[['VeloCharge']] # Wanted data from the VELO.
#    data = pd.concat([tracking, RICH, CALO, VELO], axis = 1) # Strings all the variables together into 1 dataframe.
#    return data


