
# DO NOT FORGET TO CHANGE the options file!


# Run over local data
lb-run DaVinci/v44r5 gaudirun.py options/davinci_TURCAL_pid.py data/Collision17_MagDown_Reco17_Turbo04_FULLTURBO/IOhelper.py

# Run over local MC
#lb-run DaVinci/v44r5 gaudirun.py options/davinci_MC_pid.py data/MC17_MagDown_Nu1.6-25ns-Pythia8_Sim09e_Reco17_11144001/IOhelper.py

#mv *.root output/



# Run over grid, Data
#ganga options/ganga_TURCAL_pid.py 

# Run over grid, MC
#ganga options/ganga_MC_pid.py 





