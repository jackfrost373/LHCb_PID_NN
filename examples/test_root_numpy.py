
import numpy as np
from root_numpy import array2root

# create numpy array with specific type and 'branch name'
x = np.linspace(0, 1000, num=1000, dtype=[('w', 'float64')])
print(x[15][0])

# convert array to a TTree in a TFile and write
array2root(x, 'test.root', 'test_tree', mode='recreate')
print("Wrote test.root")



from ROOT import TFile, TTree
from root_numpy import tree2array

# read the TFile using PyROOT
f = TFile.Open('test.root','READONLY')
tree = f.Get("test_tree")

# transform TTree into numpy array
x2 = tree2array(tree,['w'])
print(x2[15][0])


