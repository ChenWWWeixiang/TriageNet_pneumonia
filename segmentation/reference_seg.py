import SimpleITK as sitk 
import numpy as np
#from segmentation.lungmask import mask
import glob
from tqdm import tqdm
import os 
from predict import predict,get_model
#from segmentation.unet import UNet
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
lung_dir = '/mnt/newdisk3/seg_gz/'
leision_dir = '/mnt/newdisk2/lesion/'
root_dir = '/mnt/newdisk3/data_for2d/'
filelist = os.listdir(root_dir)
os.makedirs(leision_dir,exist_ok=True)
model2 = 'segmentation/checkpoint_final.pth'
model = get_model(model2,n_classes=2)
print('get model done')
r_count=0
imagelist = glob.glob(os.path.join(root_dir,'*','*','*.nii'))
for imagepath in tqdm(imagelist, dynamic_ncols=True):
    imagename = imagepath.split('/')[-1]
    output_path=imagepath.replace('.nii','.seg.nrrd').replace(root_dir,leision_dir)
    if os.path.exists(output_path):
        # print(imagename)
        continue
    input_image = sitk.ReadImage(imagepath)
    segmentation = predict(input_image, model = model,batch_size=8,lesion=True)
    segmentation[segmentation>1]=1
    lungpath=imagepath.replace('.nii','.nii.gz').replace(root_dir,lung_dir)
    try:
        lung_image = sitk.ReadImage(lungpath)
    except:
        r_count+=1
        continue
    lung_data = sitk.GetArrayFromImage(lung_image)
    lung_data=lung_data[:segmentation.shape[0],:segmentation.shape[1],:segmentation.shape[2]]
    try:
        leision_seg = lung_data*segmentation
        leision_seg=np.array(leision_seg,np.uint8)
        result_out= sitk.GetImageFromArray(leision_seg)
        result_out.CopyInformation(input_image)
        os.makedirs(os.path.dirname(output_path),exist_ok=True)
        sitk.WriteImage(result_out,output_path)
        print(imagename)
    except:
        r_count+=1
        continue
print(r_count)
