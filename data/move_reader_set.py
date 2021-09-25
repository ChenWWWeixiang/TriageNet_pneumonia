import os,shutil
import random
reader_root='/mnt/data11/reader_influenza_vs_covid_3'
os.makedirs(reader_root,exist_ok=True)
data=open('lists/reader_influenza_vs_covid.list','r').readlines()
random.shuffle(data)
name=[da.split(',')[0] for da in data]
for i,j in enumerate(name):
    shutil.copy(j,os.path.join(reader_root,str(i)+'.nii'))
    a=1
f=open('lists/reader_influenza_vs_covid_3.list','w')
f.writelines(data)