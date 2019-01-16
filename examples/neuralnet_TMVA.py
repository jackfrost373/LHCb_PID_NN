
import ROOT

# get the TTree
#tfile = ROOT.TFile.Open('../datafiles/output/davinci_MC_PID.root')
tfile = ROOT.TFile.Open('/home/Shared/lhcbdata/davinci_MC_PID.root')
tree = tfile.Get("PiTree/DecayTree")

# instantiate TMVA factory
fout = ROOT.TFile("tmva_out.root","RECREATE")
ROOT.TMVA.Tools.Instance()
factory = ROOT.TMVA.Factory(
    "TMVAClassification", fout,
    ":".join([ 
      "!V", 
      "!Silent", 
      "Color", 
      "DrawProgressBar",
      "Transformations=I;D;P;G;D",
      "AnalysisType=Classification"]) )

# define variables
attributesTRACK = ['TrackP','TrackPt','TrackChi2PerDof','TrackNumDof','TrackLikelihood']
attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
attributesDLLS = ['RichDLLe','RichDLLmu','RichDLLk','RichDLLp','RichDLLbt']
#attributesMuon = ['MuonBkgLL', 'MuonMuLL', 'MuonIsMuon', 'MuonNShared', 'InAccMuon', 'MuonIsLooseMuon']
attributesCALO = ['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']
attributesOther = ['VeloCharge', 'pi_TRACK_time','pi_TRACK_time_err']
#attributes = attributesTRACK + attributesRICH + attributesDLLS + attributesCALO + attributesOther
attributes = attributesDLLS

loader = ROOT.TMVA.DataLoader("dataset")
for attribute in attributes :
  loader.AddVariable(attribute,"F")
  # note: can also manipulations such as loader.AddVariable("var2 := var2 + cos(var5)")


# prepare (and split) training and testing tree. (pion=211, kaon=321, proton=2212)
sigCut = ROOT.TCut("abs(pi_TRUEID) == 321")
bkgCut = ROOT.TCut("abs(pi_TRUEID) == 211")
loader.AddSignalTree(tree)
loader.AddBackgroundTree(tree)

loader.PrepareTrainingAndTestTree( sigCut, bkgCut,
    ":".join([
      "nTrain_Signal=0",
      "nTrain_Background=0",
      "SplitMode=Random",
      "NormMode=NumEvents",
      "!V" ]) )


# define network layout
layoutString = ROOT.TString("Layout=TANH|128,TANH|128,TANH|128,LINEAR")
trainingStrategy = ROOT.TString("TrainingStrategy="
    "LearningRate=1e-2,"
    "Momentum=0.9,"
    "Repetitions=1,"
    "ConvergenceSteps=2,"
    "BatchSize=256,"
    "TestRepetitions=10,"
    "WeightDecay=1e-4,"
    "Regularization=L2,"
    "DropConfig=0.0+0.0+0.0+0.0,"
    "Multithreading=True")
dnnOptions = ROOT.TString(
    "!H:"
    "!V:"
    "ErrorStrategy=CROSSENTROPY:"
    "VarTransform=N:"
    "WeightInitialization=XAVIERUNIFORM:")
dnnOptions.Append(layoutString)
dnnOptions.Append(":")
dnnOptions.Append(trainingStrategy)

# book method (dnn)
stdOptions = dnnOptions + ROOT.TString(":Architecture=CPU")
factory.BookMethod(loader, ROOT.TMVA.Types.kDNN, "DNN", stdOptions)


# train method
factory.TrainAllMethods()

# test method
factory.TestAllMethods()
factory.EvaluateAllMethods()

# display performance
c = factory.GetROCCurve(loader)
c.Draw()




## apply trained NN on 'new' data

tree2 = tree

reader = ROOT.TMVA.Reader()
reader.BookMVA("DNN","weights/TMVAClassification_DNN.weights.xml")

#for attribute in attributes : 
#  reader.AddVariable(attribute,"F")
# TODO


