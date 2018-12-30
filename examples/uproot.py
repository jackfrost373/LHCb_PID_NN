
import uproot
tree = uproot.open("davinci_MC_PID.root")["PiTree/DecayTree"]
#tree.pandas.df(["pi_P", "pi_PT"])

