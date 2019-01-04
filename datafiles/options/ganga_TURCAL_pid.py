
# Before running on the grid, please ensure that:
# - davinci options file has no input file specified

magnet = "MagDown"
stripping = "Reco17_Turbo04"
year = "2017"

############################################################
 
# Find the right data file options from the database
dataloc = "/LHCb/Collision17/Beam6500GeV-VeloClosed-MagDown/Real Data/Reco17/Turbo04/95100000/FULLTURBO.DST"

jobname = "PID_TURCAL_{0}_{1}_{2}".format(magnet, stripping, year)
j = Job(name=jobname)
j.comment = "{0}_{1}_{2}".format(magnet, stripping, year)

# Set up the required application to run
app = "DaVinci"
version = "v44r5"
projectpath = "/project/bfys/jdevries/cmtuser"
from os import path
if not path.isdir("{0}/{1}Dev_{2}".format(projectpath,app,version)) :
  prepareGaudiExec('DaVinci','v44r5', myPath=projectpath)
myApp = GaudiExec()
j.application = GaudiExec()
j.application.directory = "{0}/{1}Dev_{2}".format(projectpath,app,version) 
j.application.options = ['./options/davinci_TURCAL_pid.py']
  
#j.backend = Local()
j.backend = Dirac()

j.outputfiles = [LocalFile('*.root'), LocalFile('stdout')]
#j.outputfiles = [DiracFile('*.root')] # stores on SE. Can download to local with j.outputfiles.get().

j.splitter = SplitByFiles(filesPerJob=5)
#j.do_auto_resubmit = True


# Get data to run over
print "Querying for data {0}".format(dataloc)
query = BKQuery(dataloc)

if not query: 
  print "Query resulted in nonetype, please check if location is correct."
  #j.remove()
else :
  j.inputdata = query.getDataset()

  j.submit()
  #queues.add(j.submit)

