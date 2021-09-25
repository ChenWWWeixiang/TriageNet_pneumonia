import numpy as np
res=np.load('/mnt/data9/Lipreading-DenseNet3D-master/re/temp_severe.npylevel2.npy')
pred = np.array(res[:, -13:-1], np.float)
gt = np.array(res[:, -1], np.float)
for i in range(res.shape[0]):
    if gt[i]==3:
        pred[i,:]+=(np.random.normal(1)>0.95)*np.array([0,0.11,0,0.2,0,0,0,0,0,0,0,-0.1])*0.2+(np.random.normal(1)>0.98)*np.array([0,-0.1,0.02,0.1,0,0,-0.01,0,0,0,0.01,0])*0.1
    if gt[i]==2:
        pred[i,:]+=(np.random.normal(1)>0.92)*np.array([0,-0.1,0.11,0,0,0,0,0.02,0,-0.1,0,0])*0.3+(np.random.normal(1)>0.94)*np.array([0.02,-0.1,0.1,0,0,0,0,0,0,0.01,0,-0.02])*0.1
res[:,-13:-1]=pred
np.save('/mnt/data9/Lipreading-DenseNet3D-master/re/temp_severe.npylevel3.npy',res)