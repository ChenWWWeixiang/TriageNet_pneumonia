
import os
import numpy as np
data=open('/mnt/data9/Lipreading-DenseNet3D-master/data/txt/croped_filted_train2.txt','r')
data_ref2=open('/mnt/data9/Lipreading-DenseNet3D-master/data/txt/pos_map_val.txt','r') 
data_ref=open('/mnt/data9/Lipreading-DenseNet3D-master/data/txt/pos_map_train.txt','r') 
allref=data_ref.readlines()
allref2=data_ref2.readlines()
allsubtype= ['CMV', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
th=[0.6,0.6,0.3,0.6,0.3,0.3,0.6,0.3,0.7,0.6,0.4,0.1]
with open('/mnt/data9/Lipreading-DenseNet3D-master/data/txt/pos_map_train2.txt','w') as f:
    count=0
    r=0
    for item in allref:
        f.writelines(item)
        r+=1
    for item in allref2:
        cls=item.split('/')[-2]
        if cls not in allsubtype:
            continue
        if np.random.normal()>th[allsubtype.index(cls)]+1:
            f.writelines(item)
            count+=1
    print(r,count)