
import ROOT

# some setup
ROOT.gStyle.SetOptStat(0)
c1 = ROOT.TCanvas()

# get the TTree
tfile = ROOT.TFile.Open('../datafiles/output/davinci_MC_PID.root')
tree = tfile.Get("PiTree/DecayTree")

# define the cuts
globalcut = "pi_P < 7000 && pi_TRACK_time_err > 0.1"
cut_pion = "abs(pi_TRUEID) == 211"
cut_proton = "abs(pi_TRUEID) == 2212"

# create the histograms with the efficient TTree::Draw
tree.Draw("pi_TRACK_time>>hist_pions(100,-3,3)", globalcut + " && " + cut_pion)
tree.Draw("pi_TRACK_time>>hist_protons(100,-3,3)", globalcut + " && " + cut_proton)
hist_pions = ROOT.gDirectory.Get("hist_pions")
hist_protons = ROOT.gDirectory.Get("hist_protons")

# Drawing options
hist_pions.SetLineColor(ROOT.kRed)
hist_pions.SetLineWidth(2)
hist_pions.SetTitle("")
hist_pions.GetXaxis().SetTitle("track time [ns]")
hist_pions.DrawNormalized()
hist_protons.SetLineColor(ROOT.kBlue)
hist_protons.SetLineWidth(2)
hist_protons.DrawNormalized("same")

leg = ROOT.TLegend(0.7,0.8,0.9,0.9)
leg.AddEntry(hist_pions,"pions","l")
leg.AddEntry(hist_protons,"protons","l")
leg.Draw("same")

c1.Update()
c1.SaveAs("track_time_pyroot.pdf")

