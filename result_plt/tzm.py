import numpy as np
filename = '/home/tzm/Model/LungX2CT/X_Diagnosis test/Xray_result.txt'
data=open(filename).readlines()
P=[]
GT=[]
PRE=[]
NN=[]
for line in data:
    path=line.split(' ')[0]
    pred_label=[float(((line.split(' ')[1]).split('[')[1]).split(',')[0]),float((line.split(' ')[2]).split(']')[0])]
    gt=np.argmax([float(((line.split(' ')[3]).split('[')[1]).split(',')[0]),float((line.split(' ')[4]).split(']')[0])])
    pred_flag=line.split(' ')[5]

    NN.append([[path]+pred_label+[gt]])

NN=np.array(NN)[:,0,:]
np.save('../re/xray.npy',NN)
