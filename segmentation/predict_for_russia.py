import numpy as np
import os,glob
from tqdm import tqdm
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from segmentation.unet import UNet
import SimpleITK as sitk
def get_model(model_path,n_classes=3,cuda=True):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.set_grad_enabled(False)
    model = UNet(n_classes=n_classes)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model
    
def predict(img, model, batch_size=4, lesion=False,cuda=True):
    if isinstance(model, str):
        model=get_model(model,3)
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    img = sitk.GetArrayFromImage(img)
    img[img<-1024]=-1024
    if lesion:
        img = (img + 1024) / 1624.
    else:
        img = img/255.
    # print(model)

    data = torch.from_numpy(img[:, np.newaxis, :, :])
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    _, H, W = img.shape
    res = np.zeros((1, H, W),dtype=np.int8)
    for i, data in enumerate(loader):
        images = data[0].to(device)
        images = images.float()
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        labels = labels.cpu().numpy()
        res = np.concatenate((res, labels), axis=0)

    return res[1:,:,:]

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    img_path = '/mnt/data9/COVID19_1110/studies'
    pred_path = '/mnt/data9/COVID19_1110/segs/lung'
    model_path = '/mnt/data9/Lipreading-DenseNet3D-master/segmentation/lung_checkpoint.pth'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    all_files = []
    for item in os.listdir(img_path):
        all_files+=glob.glob(os.path.join(img_path,item,'*.gz'))
    all_files=[i for i in all_files if i[-3:]=='.gz']
    print('# images:  ', len(all_files))

    for filename in tqdm(all_files, dynamic_ncols=True):
        if os.path.exists(os.path.join(pred_path, pred_path.split('/')[-1]+'_'+filename.split('/')[-1])):
            continue
        img=sitk.ReadImage(os.path.join(img_path, filename))
        #print(img.GetSize())
        result = predict(img, model_path,batch_size=8)
        #result[result>1]=1
        result=np.array(result,np.uint8)
        result=sitk.GetImageFromArray(result)
        sitk.WriteImage(result,os.path.join(pred_path, pred_path.split('/')[-1]+'_'+filename.split('/')[-1]))
        a=1
        #np.save(os.path.join(pred_path, 'pred_' + filename), result)


