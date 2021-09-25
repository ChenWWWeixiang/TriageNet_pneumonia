import numpy as np
from sklearn.manifold import TSNE
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2,os
x = np.load('saves/X_dbz_t.npy',allow_pickle=True).tolist()
x = np.array([np.mean(item,0) for item in x])
c=np.load('saves/Y_dbz_t.npy')
#c=c-2
ts = TSNE(n_components=2)

plt.figure(figsize=(12,12))
y = ts.fit_transform(x)
#y=np.load('saves/T_Y_dbz.npy')
#z=np.load('../saves/Z_flu.npy')
np.save('saves/T_Y_dbz.npy',y)

w=2000
h=2000
hugemap=np.ones((w,h,3))*255
posmap=np.zeros((w,h,3))
def draw_margin(I,color,line_width):
    for i in range(I.shape[1]):
        I[:line_width,  i, :] = color
        I[-line_width:, i, :] = color
    for i in range(I.shape[0]):
        I[i,:line_width, :] = color
        I[i,-line_width:, :] = color
    return I
def loadimg(path):
    data=sitk.ReadImage(path)
    data=sitk.GetArrayFromImage(data)
    L=path.replace('_data','_segs/lungs')
    dirname=path.split('/')[5]
    lung=sitk.ReadImage(L.replace(dirname+'/',dirname+'/'+dirname+'_'))
    lung=sitk.GetArrayFromImage(lung)
    idx,idy,idz=np.where(lung>0)
    data=data[idx.min():idx.max(),idy.min():idy.max(),idz.min():idz.max()]
    lens=data.shape[0]
    data=data[lens//2,:,:]
    data[data > 500] = 500
    data[data < -1200] = -1200
    data = data * 255.0 / 1700
    data = data - data.min()
    return data

cls=['CMV', 'Respiratory syncytial','Covid-19']+\
    ['Aspergillus', 'Candida']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['Chlamydia','Mycoplasma',]
m=['.','^','+','.','^','.','^','+','p','d','.','.']
cl=['b','b','b','r','r','y','y','y','y','y','m','g']
# m=['.','.','.','^','^','+','+','+','+','+','p','d']
# cl=['b','y','g','g','m','y','g','m','b','r','m','g']
import random
for i in range(len(cls)):
    plt.scatter(y[c==i, 0], y[c==i, 1],label=cls[i],marker=m[i],c=cl[i])
# for i in range(y.shape[0]):
#      if z[i][0].split('/')[-2]+'-'+z[i][0].split('/')[-1] in ai_false_case:
#          plt.scatter(y[i,0]+random.randrange(-100,100),y[i,1]+random.randrange(-100,100),c=Color[int(c[i])],marker='s',label='False'+cls[int(c[i])])
plt.title('t-SNE Curve', fontsize=16)
plt.tight_layout()
plt.legend()
plt.savefig('tSNE_dbz.jpg')
#plt.show()
