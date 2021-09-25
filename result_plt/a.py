import os,random
allsubtype= ['CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
w=[0.1,0.2,0,0.1,0.05,
    0.1,0.1,0.1,0.2,
    0,0.1,0.1,0.1,0.05,
    0.2,
    0]
fv=open('data/txt/croped_filted_val.txt','r')
ft=open('data/txt/croped_filted_train.txt','r')
alls=ft.readlines()
with open('data/txt/croped_filted_train2.txt','w') as f:
    for item in fv.readlines():
        if item.split('/')[6]=='healthy':
            continue
        i=allsubtype.index(item.split('/')[6])
        if random.random()<w[i]:
            alls.append(item)
    for item in alls:
        f.writelines(item)
