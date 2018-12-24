
hlt2lineLb = 'Hlt2PIDLb2LcMuNuTurboCalib'
hlt2lineDz = 'Hlt2PIDD02KPiTagTurboCalib'

from Configurables import DecayTreeTuple 
from DecayTreeTuple.Configuration import *

dtt_Lb = DecayTreeTuple('LcTree_Proton')
dtt_Lb.Inputs = ['{0}/Particles'.format(hlt2lineLb)]
dtt_Lb.setDescriptorTemplate( "${LambdaB0}[ Lambda_b0 -> ( Lambda_c+ -> K- ${proton}p+ pi+ ) mu- ]CC" )

dtt_Dz_k = DecayTreeTuple('DzTree_Kaon')
dtt_Dz_k.Inputs = ['{0}/Particles'.format(hlt2lineDz)]
dtt_Dz_k.setDescriptorTemplate( "${Dstar0}[ D*(2010)+ -> ( D0 -> ${Kaon}K- pi+ )  pi+ ]CC" )

dtt_Dz_pi = DecayTreeTuple('DzTree_Pion')
dtt_Dz_pi.Inputs = ['{0}/Particles'.format(hlt2lineDz)]
dtt_Dz_pi.setDescriptorTemplate( "${Dstar0}[ D*(2010)+ -> ( D0 -> K- ${pion}pi+ )  pi+ ]CC" )

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
#tupletools.append("TupleToolMCTruth")          # MC Truth information
#tupletools.append("TupleToolMCBackgroundInfo") # BKGCAT information

for i,branch in enumerate([dtt_Lb.proton, dtt_Dz_k.Kaon, dtt_Dz_pi.pion]) :
  for tt in tupletools :
    branch.addTupleTool(tt+"/"+tt+str(i))

  #hybridtool = branch.addTupleTool('LoKi::Hybrid::TupleTool')
  #hybridtool.Variables = {'ETA' : '0.5 * log( (P+PZ)/(P-PZ) )' ,
  #                        'PHI' : 'atan2(PY,PX)' }

#dtt_pi = DecayTreeTuple('PiTree')
#dtt_pi.Inputs = ['Phys/StdAllLoosePions/Particles']
#dtt_pi.Inputs = ['Turbo/pPhys/Particles']
#dtt_pi.Decay = "[pi+]CC"
#dtt_pi.addTupleTool("TupleToolTrackTime")


from Configurables import LoKi__HDRFilter as Filter
hltfilter  = Filter('HLT2Filter', Code = "HLT_PASS_RE( '{0}Decision' ) | HLT_PASS_RE( '{1}Decision' )".format(hlt2lineLb, hlt2lineDz) )

from Configurables import DaVinci
DaVinci().Turbo = True
DaVinci().InputType = 'MDST'
DaVinci().DataType = '2017'
DaVinci().RootInTES = '/Event/Turbo'
DaVinci().TupleFile = "davinci_TURCAL_PID.root"
DaVinci().HistogramFile = 'davinci_TURCAL_PID-histos.root'
DaVinci().EvtMax = 2000
DaVinci().PrintFreq = 100
DaVinci().EventPreFilters = [ hltfilter ]
DaVinci().appendToMainSequence([dtt_Lb, dtt_Dz_k, dtt_Dz_pi])

