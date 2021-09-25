import numpy as np
import os
from tqdm import tqdm
import torch,cv2,shutil,glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from segmentation.unet import UNet
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import SimpleITK as sitk
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def get_model(model_path, n_classes=3, cuda=True):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.set_grad_enabled(False)
    model = UNet(n_classes=n_classes)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def predict(img, model, batch_size=4, cuda=True):
    if isinstance(model, str):
        model = get_model(model,3)
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    #img = sitk.GetArrayFromImage(img)
    img[img < -1024] = -1024
    img = img / 255.
    # print(model)

    data = torch.from_numpy(img[np.newaxis, np.newaxis, :, :])
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    H, W = img.shape

    res = np.zeros((1, 512, 512), dtype=np.int8)
    for i, data in enumerate(loader):
        if not H == 512:
            data[0] = torch.nn.functional.interpolate(data[0], (512, 512))
        images = data[0].to(device)
        images = images.float()
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        labels = labels.cpu().numpy()
        res = np.concatenate((res, labels), axis=0)

    return res[1:, :, :]


if __name__ == '__main__':
    clss=['NCP','CP','Normal']
    model_path = './lung_checkpoint.pth'
    #model1 = get_model(model_path,3)

    print('get model done')
    for cls in clss:
        outpath = '/mnt/data9/independent_data/' + cls
        shutil.rmtree(outpath)

        os.makedirs(outpath,exist_ok=True)
        root='/mnt/data9/independent_raw/'+cls

        for patient in os.listdir(root):
            for scan in os.listdir(os.path.join(root,patient)):
                #I=[]
                for slice in os.listdir(os.path.join(root,patient,scan)):
                    if os.path.exists(os.path.join(outpath,patient,scan,slice)):
                        continue
                    img=cv2.imread(os.path.join(root,patient,scan,slice))[:,:,1]
                    img=cv2.resize(img,(512,512))
                    is_crop = np.mean(img == 0) > 0.4
                    img=img*1.0/255*1500-1400

                    if is_crop:
                        result=img>-1400
                        result = np.array(result, np.uint8)
                        #continue
                    else:
                        middle = img[100:400, 100:400]
                        mean = np.mean(middle)
                        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
                        centers = sorted(kmeans.cluster_centers_.flatten())
                        threshold = np.mean(centers)
                        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
                        #thresh_img = np.where(img <-50 , 1.0, 0.0)  # threshold the image
                        image_array = thresh_img
                        eroded = morphology.binary_closing(thresh_img, np.ones([5, 5]))
                        dilation = morphology.binary_opening(eroded, np.ones([8, 8]))
                        labels = measure.label(dilation)
                        label_vals = np.unique(labels)
                        regions = measure.regionprops(labels)  # »ñÈ¡Á¬Í¨ÇøÓò

                        good_labels = []
                        for prop in regions:
                            B = prop.bbox
                            print(B)
                            if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                                good_labels.append(prop.label)
                        mask = np.ndarray([512, 512], dtype=np.int8)
                        mask[:] = 0
                        for N in good_labels:
                            mask = mask + np.where(labels == N, 1, 0)
                        result = morphology.binary_closing(mask, np.ones([10, 10]))  # one last dilation
                        result = morphology.opening(result, np.ones([2, 2]))
                       # result = morphology.binary_erosion(result, np.ones([6, 6]))
                    img[img > 500] = 500
                    img[img < -1200] = -1200
                    img = img * 255.0 / 1700
                    img = img - img.min()
                    I=np.stack([img,result*img,result*255],-1)
                    os.makedirs(os.path.join(outpath,patient,scan),exist_ok=True)
                    img_out_path=os.path.join(outpath,patient,scan,slice)
                    cv2.imwrite(img_out_path,I)
                    print(img_out_path)
                    a=1



