import SimpleITK as sitk
import numpy as np
from segmentation.predict import predict,get_model
import os,glob
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
model_path = '/mnt/data9/Lipreading-DenseNet3D-master/segmentation/lung_checkpoint.pth'
data_root=['/mnt/data6/lung_data/lung_1st','/mnt/data6/lung_data/lung_2rd']
model=get_model(model_path,n_classes=3)
dar=['/home/xzw/Lung_Label/illPatient1/','/home/xzw/Lung_Label/illPatient2/']
def get_CI(value):
    sorted_scores=np.array(value)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return np.mean(value),confidence_lower,confidence_upper


DICE=[]
SPE=[]
SEN=[]
mind=1
re=[]
for no_dir,ad in enumerate(dar):
    for afile in glob.glob(os.path.join(ad,'*-label.nii')):
        if no_dir==1:
            temp=afile.split('/P')[-1].split('_Lung')[0]
            temp=temp.split('_')
            temp=temp[0]+'_'+temp[1]+'-'+temp[2]
            data_name = os.path.join(data_root[no_dir], temp + '.nii')
        else:
            data_name = os.path.join(data_root[no_dir], afile.split('/')[-1].split('_Lung')[0] + '.nii')

        label=sitk.ReadImage(afile)
        label=sitk.GetArrayFromImage(label)
        data = sitk.ReadImage(data_name)
        seg=predict(data, model, batch_size=4)
        seg[seg>1]=1
        dice=np.mean(label*seg)/(np.mean((label+seg)>0))
        spe=np.mean((seg==0)*(label==0))/np.mean(label==0)
        sen = np.mean((seg * label) == 1) / np.mean(label == 1)
        if dice <mind:
            mind=dice
            re=data_name
        DICE.append(dice)
        SPE.append(spe)
        SEN.append(sen)
print(get_CI(DICE))
print(get_CI(SPE))
print(get_CI(SEN))
print(mind,re)
f=open('re.txt','w')
f.writelines(str(np.mean(DICE))+str(mind)+str(re))