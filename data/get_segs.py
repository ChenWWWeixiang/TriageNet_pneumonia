import cv2,os
import numpy as np
root='/mnt/data9/independent_data/'
out='/mnt/data9/independent_cropped/'
for type in os.listdir(root):
    for person in os.listdir(os.path.join(root,type)):
        for stage in os.listdir(os.path.join(root,type,person)):
            ONES_I=[]
            Name=[]
            m1=250
            M1=250
            m2=250
            M2=250
            for slice in os.listdir(os.path.join(root,type,person,stage)):
                I=cv2.imread(os.path.join(root,type,person,stage,slice))
                S=I[:,:,2]
                idx=np.where(S==255)
                if np.sum(S==255)<800 or (idx[0].max()-idx[0].min())<200 or (idx[1].max()-idx[1].min())<300:
                    continue
                ONES_I.append(I)
                Name.append(slice)
                if idx[0].min()<m1:
                    m1=idx[0].min()
                if idx[1].min()<m2:
                    m2=idx[1].min()
                if idx[0].max()>m1:
                    M1=idx[0].max()
                if idx[1].max()>m2:
                    M2=idx[1].max()
            try:
                for I,name in zip(ONES_I,Name):
                    I=I[m1:M1,m2:M2,:]
                    os.makedirs(os.path.join(out,type,person,stage),exist_ok=True)
                    cv2.imwrite(os.path.join(out,type,person,stage,name),I)
            except:
                continue