def a():
    import uproot
    import pandas as pd
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    tfile = uproot.open('/home/Shared/lhcbdata/davinci_MC_PID.root')
    tree = tfile["PiTree/DecayTree"]
    df = tree.pandas.df()
    df['absid'] = df['pi_TRUEID'].abs()
    #print(df['absid'].value_counts())
    crit_global = (df['pi_TRACK_time_err'] > 0.1) & (df['pi_P'] > 1200)
    crit_types  = (abs(df['pi_TRUEID']) == 211) | (abs(df['pi_TRUEID']) == 321)
    dfsel = df[crit_global & crit_types]
    attributesTRACK = ['TrackP','TrackPt','TrackChi2PerDof','TrackNumDof','TrackLikelihood','TrackFitTChi2','TrackFitTNDoF','TrackFitMatchChi2',
                       'TrackGhostProbability','TrackCloneDist','TrackFitVeloChi2','TrackFitVeloNDoF',]
    attributesRICH = ['RichUsedAero', 'RichUsedR1Gas', 'RichUsedR2Gas', 'RichAboveMuThres', 'RichAboveKaThres']
    attributesDLLS = ['RichDLLe','RichDLLmu','RichDLLk','RichDLLp','RichDLLbt']
    attributesCALO = ['EcalPIDe', 'EcalPIDmu', 'HcalPIDe', 'HcalPIDmu', 'PrsPIDe', 'InAccBrem', 'BremPIDe']
    attributesOther = ['VeloCharge', 'pi_TRACK_time','pi_TRACK_time_err']
    attributes = attributesTRACK + attributesRICH + attributesDLLS + attributesCALO + attributesOther
#print(attributes)
#print(len(attributes))
    X = dfsel.loc[ :, attributes ]
    y = dfsel['absid'].astype('category')
    return x, y