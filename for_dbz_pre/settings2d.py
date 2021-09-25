import argparse,toml
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


CLS_DEFINE={
    'virus':['CMV','Coxsackie virus','H7N9','Respiratory syncytial','covid19'],
    'fungus':['aspergillus','candida','cryptococcus','PCP'],
    'bacteria':['Acinetobacter bowman','Klebsiella','Pseudomonas aeruginosa','S. aureus','Streptococcus'],
    'chlamydia':['chlamydia'],
    'mycoplasma':['mycoplasma'],
    'healthy':['healthy']
}
TYPEMAT=[[0,1,2,3,4],[5,6,7,8],[9,10,11,12,13],[14],[15]]
allsubtype= ['healthy','CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
allsubtype= ['CMV', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]

#maintype=['virus','mycoplasma','fungus','chlamydia','bacteria','healthy']
#maintype=['healthy','virus','fungus','bacteria','chlamydia','mycoplasma',]
maintype=['virus','fungus','bacteria','chlamydia','mycoplasma']
FILTERLIST1=[0,1,2,3,4] 
#FILTERLIST2=[0,1,9,11,15,16]
#FILTERLIST2=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
TYPEMAT=[[0,1,2],[3,4],[5,6,7,8,9],[10],[11]]

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--cls_list", type=str,
                    # default='re/cap_vs_covid.npy')
                    default=[0,1,2,3,4])
parser.add_argument("-m", "--m", type=int)
parser.add_argument("-n", "--n", type=int)
parser.add_argument("-k", "--k", type=int)
parser.add_argument("-g", "--gpu", type=str)
parser.add_argument("-s", "--saved_head", type=str,default='_')
parser.add_argument("-o", "--one_else", help="one_else", type=str,
                    default=None)

args = parser.parse_args()

#logfile='options_test2.5d.toml'
logfile='options_stack.toml'
#logfile='options_stack_dmap.toml'
#logfile='options_stack_covid.toml'
with open(logfile, 'r') as optionsFile:
    options = toml.loads(optionsFile.read())
    options['one_else']=args.one_else

FILTERLIST2=[0,1,2,3,4,5,6,7,8,9,10,11]
if args.cls_list:
    FILTERLIST1=[0,1,2,3,4]
if options['general']['mod']=='reader2':
    allsubtype= ['CMV', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','healthy',]
#maintype=['virus','mycoplasma','fungus','chlamydia','bacteria','healthy']
#maintype=['healthy','virus','fungus','bacteria','chlamydia','mycoplasma',]
    maintype=['virus','fungus','bacteria','chlamydia','healthy']
elif options['general']['mod']=='all':
    allsubtype= ['healthy','CAP','AB-in','COVID-19']
    maintype=['healthy','CAP','AB-in','COVID-19']
    FILTERLIST1=[0,1,2,3]
    FILTERLIST2=[0,1,2,3]

if args.n:
    options['model']['nt']=args.n
if args.m:
    options['model']['nf']=args.m
if args.k:
    options['model']['nc']=args.k
if args.gpu:
    options['general']['gpuid']=args.gpu

GPU_SETTING=options['general']['gpuid']
PRE_FORSAVE="-".join([maintype[item] for item in FILTERLIST1])
PRE_FORSAVE= args.saved_head+PRE_FORSAVE+'-'+str(options['model']['nt'])+'-'+str(options['model']['nf'])+'-'+str(options['model']['nc'])