import pandas as pd # To be able to easily manipulate the data from the .root files.
import uproot # To be able to open .root files and turn their contents into dataframes.
import numpy as np # To be able to use linspace.
import h5py # To create an empty hdf5 file to work with.

#total_data_kaon_pion_proton_hdf5 = h5py.File('/home/Shared/students/particle_data_big_kaon0_pion1_proton2.h5', 'w') # Makes an empty hdf5 file to store the pion, proton and kaon data in.

file_numbers = np.linspace(0, 99, 100) # Makes an array to access every folder in /home/Shared/lhc.data/ganga/20.
total_data_kaon_pion_proton = pd.DataFrame() # Dataframe that will be filled up with the pions, protons and kaons from all files.
for x in file_numbers: # Loops through all the files.
    number = '%.0f' % x # Makes the file number a string for the filename. Formatting done to ensure the string represents no decimals.
    file = uproot.open('/home/Shared/lhcbdata/ganga/20/'+number+'/output/davinci_MC_PID.root') # Opens one of the .root files.
    tree = file["PiTree/DecayTree"] # Accesses the tree of the root file.
    df = tree.pandas.df() # Turns the tree into a dataframe.
    
    tracking = df[['TrackP', 'TrackPt', 'TrackChi2PerDof', 'TrackLikelihood', 'TrackGhostProbability', 'TrackFitMatchChi2', 'TrackCloneDist', 'TrackFitVeloChi2', 'TrackFitVeloNDoF', 'TrackFitTChi2', 'TrackFitTNDoF', 'pi_TRACK_time', 'pi_TRACK_time_err']] # Desired tracking variables.
    RICH = df[['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres', 'RichDLLe', 'RichDLLmu', 'RichDLLmu', 'RichDLLk', 'RichDLLp', 'RichDLLbt']] # Desired RICH variables.
    CALO = df[['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']] # Desired CALO variable.
    VELO = df[['VeloCharge']] # Desired VELO variable.
    ID = df[['pi_TRUEID']] # Particle IDs, known from simulation.
    data = pd.concat([tracking, RICH, CALO, VELO, ID], axis = 1) # Strings all variables and the particle IDs together into one dataframe.
    
    data_kaon_pion_proton = data[(data.pi_TRUEID == -211) | (data.pi_TRUEID == 211) | (data.pi_TRUEID == -321) | (data.pi_TRUEID == 321) | (data.pi_TRUEID == 2212) | (data.pi_TRUEID == -2212)] # Filters the data to contain only pions, protons kaons and their antiparticles.
    # replace each ID by what we want.
    data_kaon_pion_proton = data_kaon_pion_proton.replace(to_replace = 2212, value = 2) # Protons get the ID 2.
    data_kaon_pion_proton = data_kaon_pion_proton.replace(to_replace = -2212, value = 2) # Antiprotons get ID 1.
    data_kaon_pion_proton = data_kaon_pion_proton.replace(to_replace = 211, value = 1) # Pions get the ID 1.
    data_kaon_pion_proton = data_kaon_pion_proton.replace(to_replace = -211, value = 1) # Antipions get ID 1.
    data_kaon_pion_proton = data_kaon_pion_proton.replace(to_replace = 321, value = 0) # Kaons get ID 0.
    data_kaon_pion_proton = data_kaon_pion_proton.replace(to_replace = -321, value = 0) # Antikaons get ID 0.
    total_data_kaon_pion_proton = pd.concat([total_data_kaon_pion_proton, data_kaon_pion_proton], axis = 0) # Links the data from one file to that of the files before it.
    
    #total_data_kaon_pion_proton_hdf5 = data_kaon_pion_proton.to_hdf('/home/Shared/students/particle_data_big_kaon0_pion1_proton2.h5', key = 'kaon_pion_proton', append = True, mode = 'r+', format = 'table')
    
    print('File '+number+ ': done') # To see progress during the process.

total_data_kaon_pion_proton_hdf5 = total_data_kaon_pion_proton.to_hdf('/home/Shared/students/particle_data_big_kaon0_pion1_proton2.h5', key = 'kaon_pion_proton', format = 'table')