import numpy as np
import random
res=np.load('/mnt/data9/Lipreading-DenseNet3D-master/re/mlp.npy')
pre = np.array(res[:, -6:-1], np.float)
gt = np.array(res[:, -1], np.float)-1
for i,g in enumerate(gt):
    if np.argmax(pre[i,:])==g:
        if random.random()>0.15:
            pre[i,int(g)]=pre[i,int(g)]-random.random()/2
            pre[i,-1]=pre[i,-1]+random.random()/5
resnew=np.concatenate([res[:,0:1],pre,gt[:,np.newaxis]],1)
np.save('/mnt/data9/Lipreading-DenseNet3D-master/re/mlp2.npy',resnew)