
year = "2017"
conddbtag = "sim-20180411-vc-md100" 
dddbtag = "dddb-20170721-3" 

from Configurables import DecayTreeTuple 
from DecayTreeTuple.Configuration import *

dtt_pi = DecayTreeTuple('PiTree')
dtt_pi.Inputs = ['Phys/StdAllLoosePions/Particles']
dtt_pi.setDescriptorTemplate( "${pi}[ pi+ ]CC" )

tupletools = []
#tupletools.append("TupleToolKinematic")        # Mass and momenta
#tupletools.append("TupleToolPid")              # PID info
tupletools.append("TupleToolANNPID")           # ProbNN for specific MC tunes
tupletools.append("TupleToolANNPIDTraining")   # ProbNN input variables
tupletools.append("TupleToolTrackTime")        # OT track time
#tupletools.append("TupleToolGeometry")         # ENDVERTEX, OWNPV, IP, FD, DIRA 
#tupletools.append("TupleToolAngles")           # CosTheta, angle between daughter tracks
#tupletools.append("TupleToolEventInfo")        # Runnr, eventnr, gpstime, magpol, BX
#tupletools.append("TupleToolPropertime")       # Proper lifetime TAU in ns 
#tupletools.append("TupleToolTrackInfo")        # TRACK info
#tupletools.append("TupleToolPrimaries")        # nPV, PV pos, PVnTracks
tupletools.append("TupleToolMCTruth")          # MC Truth information
#tupletools.append("TupleToolMCBackgroundInfo") # BKGCAT information

for tt in tupletools :
  tmp = dtt_pi.pi.addTupleTool(tt)

  if (tt=="TupleToolANNPID") :
    tmp.ANNPIDTunes = ["MC12TuneV4","MC15TuneV1","MC15TuneDNNV1","MC15TuneFLAT4dV1","MC15TuneCatBoostV1"]
    tmp.PIDTypes = ["Pion","Kaon","Proton"]


from Configurables import DaVinci
DaVinci().InputType = 'DST'
DaVinci().DataType = year
DaVinci().Simulation = True
DaVinci().Lumi = False
DaVinci().TupleFile = "davinci_MC_PID.root"
DaVinci().HistogramFile = 'davinci_MC_PID-histos.root'
DaVinci().EvtMax = -1
DaVinci().PrintFreq = 100
DaVinci().appendToMainSequence([dtt_pi])
DaVinci().DDDBtag  = dddbtag 
DaVinci().CondDBtag = conddbtag

