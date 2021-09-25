import json
import SimpleITK as sitk
maintype=['virus','fungus','bacteria','chlamydia','mycoplasma',]
allsubtype= ['CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
data=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/val.json','r'))
collected=dict()
with open('answer1.txt','r') as f:
    all_readef=f.readlines()
    for afile in all_readef:
        id, name,_=afile.split(',')
        for k in data.keys():
            if data[k]['newpath']==name:
                collected[k]=data[k]
json.dump(collected,open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/reader1.json','w'))
collected=dict()
with open('answer2.txt','r') as f:
    all_readef=f.readlines()
    for afile in all_readef:
        id, name,_=afile.split(',')
        for k in data.keys():
            if data[k]['newpath']==name:
                collected[k]=data[k]
json.dump(collected,open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/reader2.json','w'))