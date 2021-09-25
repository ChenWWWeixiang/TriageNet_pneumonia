import os
data=open('ind_list.list','r').readlines()
D=[]
for a in data:
    num=os.listdir(a[:-1])
    if not num==0:
        D.append(a)
with open('ind_list2.list','w') as f:
    for d in D:
        f.writelines(d)