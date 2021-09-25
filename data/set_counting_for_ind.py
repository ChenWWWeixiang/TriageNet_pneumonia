import os,glob
import numpy as np
train=True
if train:
    f = open('txt/ind_train_jpg.txt', 'r')
    data = f.readlines()
    cls = []
    data=[da.split('/')[-4] + '/' + da.split('/')[-3]+'/'+da.split('/')[-2]+'/' for da in
              data]
    data=list(set(data))
    #data = [da.split(',')[0] for da in data]
    person = [da.split('/')[-4] + '/' + da.split('/')[-3]+'/' for da in
              data]
else:
    f=open('lists/ind_list2.list','r')
    data=f.readlines()
    cls=[]
    person = [da.split('/')[-3] + '/' + da.split('/')[-2]+'/' for da in
              data]

person = list(set(person))
gender=[]
age=[]
num=[]
for data_path in person:
    if 'Normal' in data_path:
        cls.append(0)
        fullname=[da for da in data if data_path in da]
        try:
            num.append(len(fullname))
        except:
            a = 1

    elif 'NCP' in data_path:
        cls.append(1)

    else:
        cls.append(2)


nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
print('patient',nums)
age=np.array(age)
gender=np.array(gender)
num=np.array(num)
age=age//20
#print(np.sum(gender=='M'),np.sum(gender=='F'))
#print(np.sum(age==0),np.sum(age==1),np.sum(age==2),np.sum(age==3),np.sum(age>=4))
print(np.sum(num==1),np.sum(num==2),np.sum(num==3),np.sum(num>3))
print(np.sum(num==1)+2*np.sum(num==2)+3*np.sum(num==3)+np.sum(num[num>3]))