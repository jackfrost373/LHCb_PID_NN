# Example to get branches of variable length ('jagged') into a hdf5 file

import uproot, h5py, awkward

tree = uproot.open("brunelNtuple.root")["RawDataTuple/ITLiteClusters"]
array = tree.array("lastbeginX")    # can include arrays of floats per event

h5file = h5py.File("output.hdf5",'w')
akdh5 = awkward.hdf5(h5file)
akdh5["lastbeginX"] = array      # writing
h5file.close()


# open file
openfile = h5py.File("output.hdf5",'r')
openakdh5 = awkward.hdf5(openfile)

openarr = openakdh5["lastbeginX"]
print(openarr)
