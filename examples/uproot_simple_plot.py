
import uproot

# Open file
tfile = uproot.open('../datafiles/output/davinci_MC_PID.root')
tree = tfile["PiTree/DecayTree"]
#print(tree.keys())

# Fetch array
#array_PT = tree["pi_PT"].array()
vardict = tree.arrays( tree.keys() )
array_PT = vardict[b'pi_PT']

# Plot
import matplotlib.pyplot as plt
plt.hist(array_PT,100,(0,10000),color='c')
plt.show()

