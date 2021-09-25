import warnings
from torch.functional import split
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
import pickle,pdb
from PIL import Image
from radiomics import featureextractor
#import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
#import torchvision
import torch,json
import time
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip,StatefulRotate
import SimpleITK as sitk
import os
import numpy as np
import glob,six
import pandas as pd
import random
import cv2 as cv
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
from for_dbz_pre.settings2d import *

class NCPDataset(Dataset):
    def __init__(self, index_root, padding, augment=False,z_length=5):
        self.padding = padding
        self.data = []
        self.padding = padding
        self.augment = augment
        self.z_length=z_length
        with open(index_root, 'r') as f:
        #list=os.listdir(data_root)
            self.data=f.readlines()
            self.mask=[item.split(',')[-1][:-1] for item in  self.data]
            self.data = [item.split(',')[-0] for item in self.data]
        cls = []
        for data_path in self.data:
            if 'healthy' in data_path:
                cls.append(0)
            elif 'cap' in data_path:
                cls.append(1)
            else:
                cls.append(2)  # covid
        self.labels=cls
        print('num of data:', len(self.data))

    def __len__(self):
        return len(self.data)
    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weigghts for dealing with class imbalance problem

        """
        n_samples = len(self.labels)
        unique, cnts = np.unique(self.labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.labels:
            weights.append((n_samples / float(cnt_dict[label])))

        return weights
    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        if 'healthy' in data_path:
            cls = 0
        elif 'cap' in data_path:
            cls = 1
        else:
            cls = 2
        seg_path = self.mask[idx]
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)

        Mask = sitk.ReadImage(seg_path)
        M = sitk.GetArrayFromImage(Mask)

        valid=M.sum(1).sum(1)>500
        M=M[valid,:,:]
        data=data[valid,:,:]
        try:
            xx, yy, zz = np.where(M > 0)
            data = data[min(xx):max(xx), min(yy):max(yy), min(zz):max(zz)]
            M = M[min(xx):max(xx), min(yy):max(yy), min(zz):max(zz)]
        except:
            print(data_path)

        #data=np.stack([data,data,data],0)
        data[data > 500] = 500
        data[data < -1200] = -1200
        data = data * 255.0 / 1700
        data=(data+1200).astype(np.uint8)
        if self.augment:
            data,M=self.do_augmentation(data,M)
        #cv.imwrite('temp.jpg', data[:,64,:,:])
        temporalvolume,length = self.bbc(data, self.padding, self.z_length)
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':torch.LongTensor([length]),
            'features': torch.LongTensor([length])
            }
    def do_augmentation(self, array, mask):

        #array = array[None, ...]
        patch_size = np.asarray(array.shape)
        augmented = noise_transforms.augment_gaussian_noise(
            array, noise_variance=(0, .015))
        # need to become [bs, c, x, y, z] before augment_spatial
        augmented = augmented[None,None, ...]

        mask = mask[None, None, ...]
        r_range = (0, (15 / 360.) * 2 * np.pi)
        r_range2 = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.
        augmented, mask = spatial_transforms.augment_spatial(
            augmented, seg=mask, patch_size=patch_size,
            do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range2, angle_y=r_range2, angle_z=r_range,
            do_scale=True, scale=(.9, 1.1),
            border_mode_data='constant', border_cval_data=cval,
            order_data=1,
            p_el_per_sample=0.5,
            p_scale_per_sample=.5,
            p_rot_per_sample=.5,
            random_crop=False
        )
        mask = mask[0]
        augmented= (augmented[0,0, :, :, :]).astype(np.uint8)
        return augmented, mask
    def bbc(self,V, padding,z_length=3):

        temporalvolume = torch.zeros((z_length, padding, 224, 224))
        for cnt,i in enumerate(range(0,V.shape[0]-z_length,z_length)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            result=[]
            for j in range(z_length):
                result.append(transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0, 0, 0], [1, 1, 1]),
                ])(V[i+j:i+j+1,:,:])[0,:,:])
            temporalvolume[:, cnt] = torch.stack(result,0)

        #print(cnt)
        return temporalvolume,cnt

class NCP2DDataset(Dataset):
    def __init__(self, data_root,index_root, padding, augment=False):
        self.padding = padding
        self.data = []
        self.data_root = data_root
        self.padding = padding
        self.augment = augment

        with open(index_root, 'r') as f:
        #list=os.listdir(data_root)
            self.data=f.readlines()

        #for item in list:
         #   self.data.append(item)

        #print('index file:', index_root)
        print('num of data:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        cls=1-int(data_path.split('/')[-1][0]=='c')
        data=np.load(os.path.join(self.data_root, data_path))

        #data[data>400]=400
        #data[data<-1700]=-1700
        #data=data+1700
        #data=(data/data.max()*255).astype(np.uint8)
        #cv.imwrite('temp.jpg', data[:,64,:,:])
        temporalvolume,length = self.bbc(data, self.padding, self.augment)
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':torch.LongTensor([length])
            }

    def bbc(self,V, padding, augmentation=True):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        croptransform = transforms.CenterCrop((224, 224))
        if (augmentation):
            crop = StatefulRandomCrop((224, 224), (224, 224))
            flip = StatefulRandomHorizontalFlip(0.5)

            croptransform = transforms.Compose([
                crop,
                flip
            ])

        for cnt,i in enumerate(range(V.shape[0])):
            if cnt>=padding:
                break
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop((256, 256)),
                croptransform,
                transforms.ToTensor(),
                transforms.Normalize([0, 0, 0], [1, 1, 1]),
            ])(V[i,:,:,:])

            temporalvolume[:, cnt] = result

        if cnt==0:
            print(cnt)
        return temporalvolume,cnt

class NCPJPGDataset(Dataset):
    def __init__(self, data_root,index_root, padding, augment=False,cls_num=2):
        self.padding = padding
        self.data = []
        self.data_root = open(data_root,'r').readlines()
        self.text_book=[item.split('\t') for item in self.data_root]
        self.padding = padding
        self.augment = augment
        self.cls_num=cls_num
        self.train_augmentation = transforms.Compose([transforms.Resize(288),##just for abnormal detector
                                                     transforms.RandomCrop(224),
                                                     #transforms.RandomRotation(45),
                                                     transforms.RandomHorizontalFlip(0.2),
                                                     transforms.RandomVerticalFlip(0.2),
                                                     transforms.RandomAffine(45, translate=(0,0.2),fillcolor=0),

                                                     transforms.ToTensor(),
                                                     transforms.RandomErasing(p=0.1),
                                                     transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                     ])
        self.test_augmentation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                 ])
        with open(index_root, 'r') as f:
        #list=os.listdir(data_root)
            self.data=f.readlines()

        #for item in list:
         #   self.data.append(item)

        #print('index file:', index_root)
        print('num of data:', len(self.data))
        pa_id=list(set([st.split('/')[-1].split('_')[0] for st in self.data]))
        #pa_id_0=[id[0]=='c' or id[1]=='.' for id in pa_id]
        #print(np.sum(pa_id_0),len(pa_id)-np.sum(pa_id_0))
        if self.cls_num==2:
            cls = [1 - int(data_path.split('/')[-1][0] == 'c' or data_path.split('/')[-1][1]=='.' or
                           data_path.split('/')[-2]=='masked_ild') for data_path in self.data]
        elif self.cls_num==4:
            cls=[]
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'CAP' in data_path:
                    cls.append(1)
                elif 'ILD' in data_path:
                    cls.append(2)
                else:
                    cls.append(3)#covid
        elif self.cls_num==5:
            cls=[]
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'lidc' in data_path:
                    cls.append(1)
                elif 'ild' in data_path:
                    cls.append(2)
                elif 'CAP' in data_path:
                    cls.append(3)#covid
                else:
                    cls.append(4)
        nums=[np.sum(np.array(cls)==i) for i in range(max(cls)+1)]
        print(nums)
        self.nums=nums
    def get_w(self):
        S=np.sum(self.nums)
        nums=S/(self.nums)
        w=nums/np.sum(nums)
        return w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        if self.cls_num==2:
            cls=1-int(data_path.split('/')[-1][0]=='c' or data_path.split('/')[-1][1]=='.' or
                      data_path.split('/')[-2]=='masked_ild')
        elif self.cls_num==3:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif data_path.split('/')[-1][1] == '.' or data_path.split('/')[-2] == 'masked_ild':
                cls = 1
            else:
                cls = 2  # covid
        elif self.cls_num==4:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'CAP' in data_path:
                cls = 1
            elif  'ILD' in data_path:
                cls = 2  # covid
            else:
                cls=3
        elif self.cls_num==5:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'lidc'in data_path:
                cls = 1
            elif 'ild'in data_path:
                cls = 2
            elif 'CAP'in data_path:
                cls=3
            else:
                cls=4 # covid
        data=Image.open(data_path)
        age = -1
        gender = -1
        if  'lidc'in data_path or data_path.split('/')[-3] == 'reader_ex':
            age = -1
            gender = -1
        elif 'ILD' in data_path:
            temp = 'ILD/' + data_path.split('/')[-1].split('_')[0]
            for line in self.text_book:
                if line[0].split('.nii')[0] == temp:
                    age = int(line[1])
                    gender = int(line[2][:-1] == 'M')  # m 1, f 0
                    break
        elif 'CAP' in data_path :
            temp = 'CAP/' + data_path.split('/')[-1].split('_')[0]
            for line in self.text_book:
                if line[0].split('.nii')[0] == temp:
                    age = int(line[1])
                    gender = int(line[2][:-1] == 'M')  # m 1, f 0
                    break
        else:
            if data_path.split('/')[-3]=='slice_test_seg':
                if len(data_path.split('/')[-1].split('_')[1])>2:
                    a=data_path.split('/')[-1].split('c--')[-1]
                    temp='test1/'+a.split('_')[0]+'_'+a.split('_')[1]
                else:
                    a = data_path.split('/')[-1].split('c--')[-1]
                    temp='train1/'+a.split('_')[0]+'_'+a.split('_')[1]
            else:
                temp = data_path.split('/')[-2].split('_')[-1] + '/' + data_path.split('/')[-1].split('_')[0] + '_' + \
                       data_path.split('/')[-1].split('_')[1]
            for line in self.text_book:
                if line[0].split('.nii')[0] == temp:
                    age = int(line[1])
                    gender = int(line[2][:-1] == 'M')  # m 1, f 0
                    break
        if self.augment:
            data=self.train_augmentation(data)
        else:
            data=self.test_augmentation(data)
            
        return {'temporalvolume': data,
            'label': torch.LongTensor([cls]),
            'length':torch.LongTensor([1]),
            'gender':torch.LongTensor([gender]),
            'age':torch.LongTensor([age])
            }

class NCPJPGtestDataset(Dataset):
    def __init__(self, data_root, pre_lung_root,padding,lists=None,exlude_lists=True,age_list=None,cls_num=2):
        self.padding = padding
        self.cls_num=cls_num
        self.data = []
        self.text_book=None
        if isinstance(age_list,str):
            self.data_root = open(age_list, 'r').readlines()
            self.text_book = [item.split('\t') for item in self.data_root]
        self.mask=[]
        if isinstance(lists,list):
            if  not exlude_lists:
                self.data=lists
                self.mask=[item.split('_data')[0]+'_seg'+item.split('_data')[1][:-1] for item in self.data]
                self.data = [item[:-1] for item in self.data]
            else:
                if isinstance(data_root, list):
                    for r1, r2 in zip(data_root, pre_lung_root):
                        D= glob.glob(r1 + '/*.n*')
                        D=[t for t in D if not (t+'\n') in lists]
                        M= [item.split('_data')[0]+'_seg'+item.split('_data')[1] for item in D]
                        self.data+=D
                        self.mask+=M
                else:
                    D = glob.glob(data_root + '/*.n*')
                    D = [t for t in D if not (t+'\n') in lists]
                    M = [item.split('_data')[0] + '_seg' + item.split('_data')[1] for item in D]
                    self.data += D
                    self.mask += M
        else:
            if isinstance (data_root,list):
                for r1,r2 in zip(data_root,pre_lung_root):
                    self.data+=glob.glob(r1+'/*.n*')
                    self.mask+=glob.glob(r2+'/*.n*')
            else:
                self.data = glob.glob(data_root)
                self.mask=glob.glob(pre_lung_root)
        self.pre_root=pre_lung_root
        self.data_root = data_root
        self.padding = padding

        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                        transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        print('num of data:', len(self.data))

        if self.cls_num == 2:
            cls = [1 - int(data_path.split('/')[-1][0] == 'c' or data_path.index('LIDC')>-1 or
                           data_path.index('ILD')>-1) for data_path in self.data]
        elif self.cls_num == 4:
            cls = []
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'CAP' in data_path:
                    cls.append(1)
                elif 'ILD' in data_path:
                    cls.append(2)
                else:
                    cls.append(3)  # covid
        #cls=0
        elif self.cls_num==5:
            cls = []
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'LIDC' in data_path:
                    cls.append(1)
                elif 'ILD' in data_path:
                    cls.append(2)
                elif 'CAP' in data_path:
                    cls.append(3)
                else:
                    cls.append(4)  # covid
        nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
        print(nums)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        #print(data_path,mask_path)
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        if self.cls_num == 2:
            cls = 1 - int(data_path.split('/')[-1][0] == 'c' or data_path.split('/')[-1][1] == '.' or
                          data_path.split('/')[-3] == 'ILD')
        elif self.cls_num == 3:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif data_path.split('/')[-1][1] == '.' or data_path.split('/')[-3] == 'ILD':
                cls = 1
            else:
                cls = 2  # covid
        elif self.cls_num==4:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'CAP' in data_path:
                cls = 1
            elif 'ILD' in data_path:
                cls = 2  # covid
            else:
                cls = 3
        elif self.cls_num==5:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'LIDC' in data_path:
                cls = 1
            elif 'ILD' in data_path:
                cls = 2
            elif 'CAP' in data_path:
                cls = 3
            else:
                cls = 4# covid

        mask = sitk.ReadImage(mask_path)
        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        M=M[:data.shape[0],:,:]
        valid=np.where(M.sum(1).sum(1)>500)
        data = data[valid[0], :, :]
        M = M[valid[0], :data.shape[1], :data.shape[2]]
        data=data[:M.shape[0],:M.shape[1],:M.shape[2]]
        temporalvolume,name = self.bbc(data, self.padding,M)
        age = -1
        gender = -1

        if isinstance(self.text_book,list):
            if 'LIDC' in data_path or\
                  data_path.split('/')[-3] == 'reader_ex':
                age = -1
                gender = -1
            elif 'ILD' in data_path:
                temp = 'ILD/' + data_path.split('/')[-1].split('.nii')[0]
                for line in self.text_book:
                    if line[0].split('.nii')[0] == temp:
                        age = int(line[1])
                        try:
                            gender = int(line[2][:-1] == 'M')  # m 1, f 0
                        except:
                            gender=-1
                        break
            elif 'CAP' in data_path:
                temp = 'CAP/' + data_path.split('/')[-1].split('_')[1]
                for line in self.text_book:
                    if line[0].split('.nii')[0] == temp:
                        age = int(line[1])
                        gender = int(line[2][:-1] == 'M')  # m 1, f 0
                        break
            else:
                temp = data_path.split('/')[-2].split('_')[-1] + '/' + data_path.split('/')[-1].split('_')[0] + '_' + \
                       data_path.split('/')[-1].split('_')[1]
                for line in self.text_book:
                    if line[0] == temp:
                        age = int(line[1])
                        gender = int(line[2][:-1] == 'M')  # m 1, f 0
                        break

        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':[data_path,name],
            'gender': torch.LongTensor([gender]),
            'age': torch.LongTensor([age])

            }

    def bbc(self,V, padding,pre=None):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        #croptransform = transforms.CenterCrop((224, 224))
        cnt=0
        name=[]
        for cnt,i in enumerate(range(1,V.shape[0]-1,3)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            data=V[i-1:i+1,:,:]
            data[data > 700] = 700
            data[data < -1200] = -1200
            data = data * 255.0 / 1900
            name.append(i)
            data = data - data.min()
            data = np.concatenate([pre[i-1:i, :, :] * 255,data], 0)  # mask one channel
            data = data.astype(np.uint8)
            data=Image.fromarray(data.transpose(1,2,0))
            #data.save('temp.jpg')
            result = self.transform(data)

            temporalvolume[:, cnt] = result

        if cnt==0:
            print(cnt)
        return temporalvolume,name

class NCPJPGtestDataset_MHA(Dataset):
    def __init__(self, data_root, pre_lung_root,padding,lists=None,exlude_lists=True):
        self.padding = padding
        self.data = []
        self.mask=[]
        if isinstance(lists,list):
            if  not exlude_lists:
                self.data=lists
                self.mask=[item.split('images')[0]+'lungsegs'+item.split('images')[1][:-1] for item in self.data]
                self.data = [item[:-1] for item in self.data]
            else:
                if isinstance(data_root, list):
                    for r1, r2 in zip(data_root, pre_lung_root):
                        D= glob.glob(r1 + '/*/*.mha')
                        D=[t for t in D if not (t+'\n') in lists]
                        M= [item.split('images')[0]+'lungsegs'+item.split('images')[1] for item in D]
                        self.data+=D
                        self.mask+=M
                else:
                    D = glob.glob(data_root + '/*/*.mha')
                    D = [t for t in D if not (t+'\n') in lists]
                    M = [item.split('images')[0] + 'lungsegs' + item.split('images')[1] for item in D]
                    self.data += D
                    self.mask += M
        else:
            if isinstance (data_root,list):
                for r1,r2 in zip(data_root,pre_lung_root):
                    self.data+=glob.glob(r1+'/*/*.mha')
                    self.mask+=glob.glob(r2+'/*/*.mha')
            else:
                self.data = glob.glob(data_root)
                self.mask=glob.glob(pre_lung_root)
        self.data=list(set(self.data))
        self.mask = list(set(self.mask))
        self.pre_root=pre_lung_root
        self.data_root = data_root
        self.padding = padding

        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                        transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        print('num of data:', len(self.data))

        cls = [1 - int(data_path.split('/')[-1][0] == 'c' or data_path.split('/')[-1][1] == '.'
                       or data_path.split('/')[-3]=='ILD' or data_path.split('/')[-3]=='reader_ex') for
               data_path in self.data]
        #cls=0
        print(np.sum(np.array(cls) == 0), np.sum(np.array(cls) == 1))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        cls=1-int(data_path.split('/')[-1][0]=='c'or
                  data_path.split('/')[-3]=='ILD' or
                  data_path.split('/')[-3] == 'LIDC' or
                  data_path.split('/')[-3] == 'reader_ex')

        #cls=0
        #cls=0
        #volume = sitk.ReadImage(os.path.join(input_path, name))
        mask = sitk.ReadImage(mask_path)
        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        #data = data[-300:-40, :, :]
        #print(M.shape)
        M = M[:, :data.shape[1], :data.shape[2]]
        data=data[:M.shape[0],:M.shape[1],:M.shape[2]]
        temporalvolume,name = self.bbc(data, self.padding,M)
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':[data_path,name]
            }

    def bbc(self,V, padding,pre=None):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        #croptransform = transforms.CenterCrop((224, 224))
        cnt=0
        name=[]
        for cnt,i in enumerate(range(V.shape[0]-1)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            #if cnt>=padding:
            #    break
            data=V[i:i+1,:,:]
            data[data > 700] = 700
            data[data < -1200] = -1200
            data = data * 255.0 / 1900
            name.append(i)
            data = data - data.min()
            data = np.concatenate([pre[i:i + 1, :, :] * 255,data,data], 0)  # mask one channel
            data = data.astype(np.uint8)
            data=Image.fromarray(data.transpose(1,2,0))
            #data.save('temp.jpg')
            result = self.transform(data)

            temporalvolume[:, cnt] = result

        #if cnt==0:
        print(cnt)
        temporalvolume=temporalvolume[:,:cnt+1]
        return temporalvolume,name

class NCPJPGDataset_new(Dataset):
    def __init__(self, data_root,index_root, padding, augment=False,cls_num=2,mod='ab',options=None):
        self.mod=mod
        self.padding = padding
        self.data = []
        self.options=options

        self.data_root=data_root
        self.padding = padding
        self.augment = augment
        self.cls_num=cls_num
        self.use_rc=options['use_rc']
        self.train_augmentation = transforms.Compose([transforms.Resize((224,224)),##just for abnormal detector
                                                     #transforms.RandomRotation(45),
                                                     transforms.RandomAffine(20,fillcolor=0),
                                                     #transforms.RandomCrop(224),
                                                     transforms.ColorJitter(brightness=0.5,contrast=0.5),

                                                     transforms.ToTensor(),
                                                     transforms.RandomErasing(p=0.1),

                                                     transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                     ])
        self.test_augmentation = transforms.Compose([transforms.Resize((224,224)),
                                                 #transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                 ])
        with open(index_root, 'r') as f:
            self.data=f.readlines()
        print('num of data:', len(self.data))

        if self.cls_num==2:
            if self.mod=='slice':#abnormal detection
                cls = [int('abnor' in data_path) for data_path in self.data]
            elif self.mod=='co':
                cls = [1-int('cap' in data_path) for data_path in self.data]
            else:
                cls = [1 - int('healthy' in data_path ) for data_path in self.data]
        elif self.cls_num==3:
            cls=[]
            for data_path in self.data:
                if 'Normal' in data_path:
                    cls.append(0)
                elif not 'NCP' in data_path:
                    cls.append(1)
                else:
                    cls.append(2)
        else:
            cls=[]
            for data_path in self.data:
                if self.mod == 'slice':  # abnormal detection
                    cls .append(int('abnor' in data_path)*3)
                else:
                    if 'healthy' in data_path:
                        cls.append(0)
                    elif 'cap' in data_path or 'CAP' in data_path:
                        cls.append(1)
                    elif 'AB-in' in data_path:
                        cls.append(2)#covid
                    else:
                        cls.append(3)
        nums=[np.sum(np.array(cls)==i) for i in range(self.cls_num)]
        print(nums)
        self.labels=cls
        self.nums=nums
    def get_w(self):
        S=np.sum(self.nums)
        nums=(S/(self.nums))
        w=nums/np.sum(nums)
        return w
    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weigghts for dealing with class imbalance problem

        """
        n_samples = len(self.labels)
        unique, cnts = np.unique(self.labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.labels:
            weights.append((n_samples / float(cnt_dict[label])))

        return weights
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        feature=0
        if self.cls_num==2:
            if self.mod=='slice':#abnormal detection
                cls = int('abnor' in data_path)
            elif self.mod=='co':
                cls = 1-int('ild' in data_path or 'cap' in data_path)
            else:
                cls = 1 - int('healthy' in data_path )
        elif self.cls_num==3:
            if 'Normal' in data_path:
                cls = 0
            elif not 'NCP' in data_path :
                cls = 1
            else:
                cls=2
        elif self.cls_num==4:
            if self.mod=='slice':#abnormal detection
                cls = int('abnor' in data_path)*3
            else:
                if 'healthy' in data_path:
                    cls = 0
                elif 'cap' in data_path or 'CAP' in data_path:
                    cls = 1
                elif 'AB-in' in data_path:
                    cls=2
                else:
                    cls=3
        data=Image.open(data_path)
        if not self.mod=='ind' and not self.mod=='slice' and False:
            age = int(data_path.split('_')[-3])
            gender = int(data_path.split('_')[-2]=='M')
            pos=int(data_path.split('_')[-1].split('.')[0])
        else:
            age=-1
            gender=-1
            pos=-1

        data = data.convert("RGB")
        #data.save('temp_train.jpg')

        if self.augment:
            data=self.train_augmentation(data)
        else:
            data=self.test_augmentation(data)
        return {'temporalvolume': data,
            'label': torch.LongTensor([cls]),
            'label2': torch.LongTensor([cls]),
            'length':torch.LongTensor([1]),
            'gender':torch.LongTensor([gender]),
            'age':torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos/100]),
            'name':data_path,
            'features':torch.FloatTensor([feature])
            }

class NCPJPGtestDataset_new(Dataset):
    def __init__(self, data_root,padding,lists,age_list=None,cls_num=2,mod='ab',options=None):
        #self.padding = padding
        self.data_root=data_root
        if data_root[-3:]=='csv':
            self.r=pd.read_csv(data_root)
        self.options = options
        if 'radiomics_path' in options['general'].keys():
            self.radiomics_path = options['general']['radiomics_path']
            os.makedirs(self.radiomics_path, exist_ok=True)
        else:
            self.radiomics_path = []
        self.extractor = featureextractor.RadiomicsFeatureExtractor('radiomics/RadiomicsParams.yaml')
        self.cls_num=cls_num
        self.data = []
        self.text_book=None
        self.mod=mod
        self.data=open(lists,'r').readlines()
        self.mask=[item.split(',')[1] for item in self.data]
        self.data = [item.split(',')[0] for item in self.data]
        self.padding = padding

        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         #transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        print('num of data:', len(self.data))
        person=[da.split('/')[-2]+'_'+da.split('/')[-1].split('_')[0]+'_'+da.split('/')[-1].split('_')[1] for da in self.data]
        #person = [da.split('/')[-2] + '_' + da.split('/')[-1].split('_')[0]  for
        #          da in self.data]
        person=list(set(person))


        if self.cls_num==2:
            cls = []
            for data_path in person:
                if 'healthy' in data_path or 'Normal' in data_path:
                    cls.append(0)
                else:
                    cls.append(1)  # covid
            cls_stage = []
            for data_path in self.data:
                if 'healthy' in data_path or 'Normal' in data_path:
                    cls_stage.append(0)
                else:
                    cls_stage.append(1)  # covid
        elif self.cls_num == 3:
            cls = []
            for data_path in person:
                if 'healthy' in data_path or 'Normal' in data_path:
                    cls.append(0)
                elif 'cap' in data_path or not 'NCP' in data_path:
                    cls.append(1)
                else:
                    cls.append(2)  # covid
            cls_stage=[]
            for data_path in self.data:
                if 'healthy' in data_path or 'Normal' in data_path:
                    cls_stage.append(0)
                elif 'cap' in data_path or not 'NCP' in data_path:
                    cls_stage.append(1)
                else:
                    cls_stage.append(2)  # covid
        else:
            if self.mod=='mosmed':
                cls = []
                cls_stage = []
                for data_path in person:
                    id = int(data_path.split('/')[-1].split('_')[2].split('.')[0])
                    if id < 255:
                        cls.append(0)
                        cls_stage.append(0)
                    else:
                        cls.append(3)
                        cls_stage.append(3)

            else:
                cls = []
                for data_path in person:
                    if 'healthy' in data_path:
                        cls.append(0)
                    elif 'cap' in data_path or 'CAP' in data_path:
                        cls.append(1)
                    elif 'AB-in' in data_path:
                        cls.append(2)  # covid
                    else:
                        cls.append(3)
                cls_stage = []
                for data_path in self.data:
                    if 'healthy' in data_path :
                        cls_stage.append(0)
                    elif 'cap' in data_path or 'CAP' in data_path:
                        cls_stage.append(1)
                    elif 'AB-in' in data_path:
                        cls_stage.append(2)  # covid
                    else:
                        cls_stage.append(3)
        if True:
            nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
            print('patient',nums)
            nums = [np.sum(np.array(cls_stage) == i) for i in range(np.max(cls_stage) + 1)]
            print('stages', nums)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        #print(data_path,mask_path)
        if data_path[-1]=='\n':
            data_path=data_path[:-1]

        if self.cls_num == 2:
            if self.mod=='ab':#abnormal detection
                cls = int('pos' in data_path)
            elif self.mod=='co':
                cls = 1-int('cap' in data_path)
            elif self.mod == 'mosmed':
                id=int(data_path.split('/')[-1].split('_')[1].split('.')[0])
                if id<255 :
                    cls = 0
                else:
                    cls=1
            else:
                cls = 1 - int('healthy' in data_path)
        elif self.cls_num==3:
            if self.mod == 'mosmed':
                id=int(data_path.split('/')[-1].split('_')[1].split('.')[0])
                if id<255 :
                    cls = 0
                else:
                    cls=2
            else:
                if 'healthy' in data_path or 'Normal' in data_path:
                    cls = 0
                elif 'cap' in data_path or not 'NCP' in data_path:
                    cls = 1
                else:
                    cls = 2
        else:
            if self.mod=='mosmed':
                id=int(data_path.split('/')[-1].split('_')[1].split('.')[0])
                if id<255 :
                    cls = 0
                else:
                    cls=3
            else:
                if 'healthy' in data_path :
                    cls = 0
                elif 'cap' in data_path or 'CAP' in data_path:
                    cls = 1
                elif 'AB-in' in data_path:
                    cls = 2
                else:
                    cls=3
        try:
            mask = sitk.ReadImage(mask_path[:-1])
        except:
            mask_path=mask_path.split('2020')[0]+mask_path.split('2020')[1][:-1]
            mask = sitk.ReadImage(mask_path)

        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        M=M[:data.shape[0],:,:]
        if M.max()==0:
            area = np.where(data >= 0)
        else:
            M[M>1]=1
            valid = np.where(M.sum(1).sum(1) > 200)
            if len(valid) == 0:
                area = np.where(M >= 0)
            else:
                data = data[valid[0],:,:]
                M = M[valid[0],:,:]
                area = np.where(M>0)
        try:
            data = data[:, area[1].min():area[1].max(), area[2].min():area[2].max()]
        except:
            a=1
       # L = L[valid[0], area[1].min():area[1].max(), area[2].min():area[2].max()]
        M = M[:, area[1].min():area[1].max(), area[2].min():area[2].max()]
        data=data[:M.shape[0],:M.shape[1],:M.shape[2]]
        temporalvolume,pos,feature = self.bbc(data, self.padding,data_path,M)
        try:
            age = int(data_path.split('_')[-2])
            gender = int(data_path.split('_')[-1].split('.nii')[0]=='M')
        except:
            age=-1
            gender=-1
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'label2': torch.LongTensor([cls]),
            'length':[data_path,pos],
            'gender': torch.LongTensor([gender]),
            'age': torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos]),
            'features':torch.FloatTensor(feature)
            }

    def bbc(self,V, padding,data_path,pre=None,L=None):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        F=np.zeros((padding,479))

        stride=max(V.shape[0]//padding,1)
        cnt=0
        name=[]
        for cnt,i in enumerate(range(0,V.shape[0],stride)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            data=V[i,:,:]
            data[data > 500] = 500
            data[data < -1200] = -1200

            name.append(float(i/V.shape[0]))
            data = data +1200
            data = data*255.0 /1700
            data = np.stack([pre[i,:,:]*255,pre[i, :, :] * data,data], -1)  # mask one channel
            #data = np.stack([data,data, data], -1)
            #data=cv.flip(data,0)
            #data = np.stack([pre[i, :, :] * data, pre[i, :, :] * data, pre[i, :, :] *data], -1)
            data = data.astype(np.uint8)
            data=Image.fromarray(data)
            #data.save('temp.jpg')
            #data.save('temp.jpg')
            result = self.transform(data)
            temporalvolume[:, cnt] = result

        #temporalvolume=temporalvolume
        return temporalvolume[:,:cnt,:,:],cnt,F[:cnt]
class IndtestDataset(Dataset):
    def __init__(self, data_root,padding,lists,cls_num=2,mod='ind',options=None):
        self.data_root=data_root
        self.options = options
        self.cls_num=cls_num
        self.data = []
        self.text_book=None
        self.mod=mod
        self.data=open(lists,'r').readlines()
        self.padding = padding
        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         #transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        print('num of data:', len(self.data))
        person=[da.split('/')[-2] for da in self.data]
        person=list(set(person))
        cls=[]
        for data_path in person:
            if 'healthy' in data_path or 'Normal' in data_path:
                cls.append(0)
            elif 'cap' in data_path or not 'NCP' in data_path:
                cls.append(1)
            else:
                cls.append(2)  # covid
        cls_stage=[]
        for data_path in self.data:
            if 'healthy' in data_path or 'Normal' in data_path:
                cls_stage.append(0)
            elif 'cap' in data_path or not 'NCP' in data_path:
                cls_stage.append(1)
            else:
                cls_stage.append(2)  # covid
        if not self.cls_num == 2:
            nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
            print('patient',nums)
            nums = [np.sum(np.array(cls_stage) == i) for i in range(np.max(cls_stage) + 1)]
            print('stages', nums)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]

        if data_path[-1]=='\n':
            data_path=data_path[:-1]

        if 'healthy' in data_path or 'Normal' in data_path:
            cls = 0
        elif 'cap' in data_path or not 'NCP' in data_path:
            cls = 1
        else:
            cls = 2

        all_jpgs=glob.glob(data_path+'/*.*')
        all_jpgs.sort()
        temporalvolume = torch.zeros((3, 100, 224, 224))
       # if len(all_jpgs)>400:
       #     ll=len(all_jpgs)
       #     all_jpgs=all_jpgs[ll//4:ll*3//4]
        if len(all_jpgs)>=100:
            stride=max(len(all_jpgs)//100+1,2)
            idx=np.arange(0,len(all_jpgs),stride)
            all_jpgs=np.array(all_jpgs)
            all_jpgs=all_jpgs[idx].tolist()
        cnt=0
        for cnt,one_jpg in enumerate(all_jpgs):
            if cnt==100:
                break
            data=Image.open(one_jpg)

            data = data.convert("RGB")
            data.save('temp_test.jpg')
            result = self.transform(data)
            #lung_seg=result[0,:,:]
            temporalvolume[:, cnt] = result
        if cnt==0:
            a=1
        temporalvolume=temporalvolume[:,:cnt,:,:]
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':[data_path,cnt],
            }

class DBZ_dataset(Dataset):
    def __init__(self, data_root,index_root, padding, augment=False,cls_num=6,mod='ab',options=None,logger=None,one_else=None):
        self.mod=mod
        self.padding = padding
        self.data = []
        self.rcdata=pickle.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/all_r2.pkl','rb'))
        # np.array(X),np.array(label1),np.array(label2), np.array(name)
        #self.cdata=json.load(open(self.cdata,'r'))
        self.options=options
        self.logger=logger
        #self.data_root=data_root
        #self.padding = padding
        self.augment = augment
        self.cls_num=cls_num
        self.infofile=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/clinical_all.json','r'))
        self.train_augmentation = transforms.Compose([transforms.Resize((256,256)),##just for abnormal detector
                                                     #transforms.RandomRotation(45),
                                                     transforms.RandomAffine(20,fillcolor=0),
                                                     transforms.RandomCrop(224),
                                                     transforms.ColorJitter(brightness=0.5,contrast=0.5),
                                                     transforms.ToTensor(),
                                                     #transforms.Er(p=0.1),

                                                     transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                     ])
        self.test_augmentation = transforms.Compose([transforms.Resize((256,256)),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                 ])

        with open(index_root, 'r') as f:
            self.data=f.readlines()
        self.logger.info('num of data:'+str(len(self.data)))

        clsI=[]
        clsII=[]
        Data=[]
        info=[]
        if one_else:
            level=int(one_else[0])
            ff=int(one_else[1:])
        for data_path in self.data:
            type1=data_path.split('/')[-3]
            type2=data_path.split('/')[-2]
            if not type1 in maintype:
                continue
            t=maintype.index(type1)
            try:
                t2=allsubtype.index(type2)
            except:
                continue
            if t in FILTERLIST1:
                if t2 in FILTERLIST2:
                    if options['general']['clinic']:
                        id=data_path.split('/')[-1].split('_')[0]
                        if id in self.infofile.keys():
                            info.append(self.infofile[id]['clinic_f'])
                        else:
                            continue
                    if one_else:
                        if level==1:
                            clsI.append(t==ff)
                            clsII.append(t2)
                        else:
                            clsI.append(t)
                            clsII.append(t2==ff)
                    else:
                        clsI.append(t)
                        clsII.append(t2)
                    Data.append(data_path)
        types, nums = np.unique(clsI,return_counts=True)
        types2, nums2 = np.unique(clsII, return_counts=True)
        self.logger.info('{} dataset'.format(index_root))
        self.logger.info('for '+ str(maintype)+'num is :'+str(nums))
        self.logger.info('for '+str(allsubtype)+'num is :'+str(nums2))
        print('{} dataset'.format(index_root))
        print('for '+ str(maintype)+'num is :'+str(nums))
        print('for '+str(allsubtype)+'num is :'+str(nums2))
        self.labels=clsI
        self.labels2=clsII
        self.data=Data
        self.info=info

    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weights for dealing with class imbalance problem

        """
        n_samples = len(self.labels2)
        unique, cnts = np.unique(self.labels2, return_counts=True)
        #cnts=cnts
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.labels2:
            weights.append((n_samples / float(cnt_dict[label])**1))

        return weights
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        while(1):
            #print(idx)
            data_path = self.data[idx]
            if data_path[-1]=='\n':
                data_path=data_path[:-1]         
            cls=self.labels[idx]
            cls2=self.labels2[idx]
            
            if self.options['use_rc']:
                try:
                    pid=data_path.split('/')[-1].split('_')[0]
                    infos=self.infofile[pid]
                   # pdb.set_trace()
                    name=infos['name']
                    ii=self.rcdata[-1].tolist().index(name)
                    #cfeatures=np.zeros(8)
                    cfeatures=self.rcdata[0][ii,:][-8:].astype(np.float)
                    rfeatures=self.rcdata[0][ii,:].astype(np.float)
                    rfeatures[np.isnan(rfeatures)]=0
                except:
                    cfeatures=np.zeros(8)
                    rfeatures=np.zeros(1321)
            else:
                rfeatures,cfeatures=0,0
            if self.options['general']['mc']:
                data=Image.open(data_path)
                info=data_path.split('/')[-1].split(':')[0].split('_')
                age,gender,pos=info[-3],info[-2],int(info[-1])
                gender=gender=='M'
                try:
                    age=int(age)
                except:
                    age=int(age[:-1])
                data = data.convert("RGB")
                temporalvolume = torch.zeros( (15, 224, 224))
                if self.augment:
                    data=self.train_augmentation(data)
                else:
                    data=self.test_augmentation(data)
                temporalvolume[0,:,:] = data[0,:,:]
                for i in range(14):
                    extra_path=data_path.replace(':0.jpg',':{}.jpg'.format(i))
                    e_data=Image.open(extra_path)
                    e_data = e_data.convert("RGB")
                    if self.augment:
                        e_data=self.train_augmentation(e_data)
                    else:
                        e_data=self.test_augmentation(e_data)
                    temporalvolume[i+1,:,:] = e_data[0,:,:]
                data = temporalvolume
            else:
                data=Image.open(data_path)
                info=data_path.split('_')
                age,gender,pos=info[-3],info[-2],int(info[-1][:-4])
                try:
                    age=int(age)
                except:
                    age=int(age[:-1])
                gender=gender=='M'
                data = data.convert("RGB")
                #data.save('temp_train.jpg')
                if self.augment:
                    data=self.train_augmentation(data)
                else:
                    data=self.test_augmentation(data)
            break
        return {'temporalvolume': data,
            'label': torch.LongTensor([cls]),
            'length': torch.LongTensor([1]),
            'label2':torch.LongTensor([cls2]),
            'gender':torch.LongTensor([gender]),
            'age':torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos/100]),
            'name':data_path,
            'rfeature':torch.FloatTensor([rfeatures]),
            'cfeature':torch.FloatTensor([cfeatures])
            }


class DBZ_test_dataset(Dataset):
    def __init__(self, data_root,padding,lists,age_list=None,cls_num=6,mod='ab',options=None,logger=None,one_else=None):
        #self.padding = padding
        #self.data_root=data_root
        #if data_root[-3:]=='csv':
        #    self.r=pd.read_csv(data_root)
        self.filter='all'
        self.options = options
        self.rcdata=pickle.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/all_r2.pkl','rb'))
        self.cls_num=cls_num
        self.data = []
        self.mask=[]
        self.clsI=[]
        self.clsII = []
        self.cc=[]
        self.text_book=None
        self.mod='dbz'
        filter_list=open('data/clsuter.txt','r').readlines()
        severe_list=[a.split('\t')[0] for a in filter_list]
        id_list=[a.split('\t')[1][3:-5] for a in filter_list]
        self.logger=logger
        self.infos=[]
        info=[]
        self.raw_data=json.load(open(lists,'r'))
        self.infofile=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/clinical_all.json','r'))
        if one_else:
            level=int(one_else[0])
            ff=int(one_else[1:])
        for one in self.raw_data.keys():
            aa=1
            thisone=self.raw_data[one]
            pid=thisone['pid']
            if self.filter in ['mild','severe']:
                if not pid in id_list:
                    continue
                if not severe_list[id_list.index(pid)]==self.filter:
                    continue
            if not thisone['clsI'] in maintype:
                continue
            clsI=maintype.index(thisone['clsI'])
            try:
                clsII=allsubtype.index(thisone['clsII'])
            except:
                continue
            if clsI in FILTERLIST1:
                if clsII  in FILTERLIST2:
                    if options['general']['clinic']:
                        id=thisone['newpath'].split('/')[-1].split('-')[0]
                        if id in self.infofile.keys():
                            info.append(self.infofile[id]['clinic_f'])
                        else:
                            continue
                    self.data.append(thisone['newpath'])
                    self.mask.append(thisone['newpath'].replace('data_for2d', 'seg_for2d'))
                    self.cc.append(thisone['newpath'].replace('newdisk3/data_for2d','newdisk2/reg/transform').replace('.nii','.seg.nii'))

                    if one_else:
                        if level==1:
                            self.clsI.append(clsI==ff)
                            self.clsII.append(clsII)
                        else:
                            self.clsI.append(clsI)
                            self.clsII.append(clsII==ff)
                    else:
                        self.clsI.append(clsI)
                        self.clsII.append(clsII)
                    self.infos.append(thisone)
        self.padding = padding
        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         #transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        
        self.logger.info('{} dataset'.format(lists))


        nums = [np.sum(np.array(self.clsI) == i) for i in range(np.max(self.clsI) + 1)]
        self.logger.info('maintype'+str(nums))
        print('{} dataset'.format(lists))
        print('for maintype num is :'+str(nums))
        nums = [np.sum(np.array(self.clsII) == i) for i in range(np.max(self.clsII) + 1)]
        self.logger.info('subtype'+str(nums))
        print('for subtype num is :'+str(nums))
        self.info=info
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        rawinfo=self.infos[idx]
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        direct=False
        try:
            name=rawinfo['name']
        except:
            name=0
        #print(data_path,mask_path)
        cls=self.clsI[idx]
        clsII = self.clsII[idx]
        if self.options['use_rc']:
            try:
                ii=self.rcdata[-1].tolist().index(name)
                #cfeatures=np.zeros(8)
                cfeatures=self.rcdata[0][ii,:][-8:].astype(np.float)
                rfeatures=self.rcdata[0][ii,:].astype(np.float)
                rfeatures[np.isnan(rfeatures)]=0
            except:
                cfeatures=np.zeros(8)
                rfeatures=np.zeros(1321)
        else:
            rfeatures,cfeatures=0,0
        if self.options['general']['mc']:
            cc_path=self.cc[idx]
            cc = sitk.ReadImage(cc_path)  
            cc_ar=sitk.GetArrayFromImage(cc)
            mask = sitk.ReadImage(mask_path)
            M = sitk.GetArrayFromImage(mask)
            volume=sitk.ReadImage(data_path)
            data=sitk.GetArrayFromImage(volume)

            M=M[:data.shape[0],:,:]
            if M.max()==0:
                area = np.where(data >= 0)
            else:
                M[M>1]=1
                valid = np.where(M.sum(1).sum(1) > 800)
                if len(valid) == 0:
                    area = np.where(M >= 0)
                else:
                    data = data[valid[0],:,:]
                    M = M[valid[0],:,:]
                    cc_ar=cc_ar[valid[0],:,:]
                    area = np.where(M>0)
            temporalvolume,pos,_ = self.bbc(data, self.padding,data_path,M,cc_ar)
            try:
                age = int(rawinfo['age'])
                gender = int(rawinfo['gender']=='M')
            except:
                age=int(rawinfo['age'][:-1])
                gender=int(rawinfo['gender']=='M')
        else:
            if not direct:
                mask = sitk.ReadImage(mask_path)
                M = sitk.GetArrayFromImage(mask)
                volume=sitk.ReadImage(data_path)
                data=sitk.GetArrayFromImage(volume)
                M=M[:data.shape[0],:,:]
                if M.max()==0:
                    area = np.where(data >= 0)
                else:
                    M[M>1]=1
                    valid = np.where(M.sum(1).sum(1) > 500)
                    if len(valid) == 0:
                        area = np.where(M >= 0)
                    else:
                        data = data[valid[0],:,:]
                        M = M[valid[0],:,:]
                        area = np.where(M>0)
                if self.options['input']['croped']:
                        M = M[:,area[1].min():area[1].max(),area[2].min():area[2].max()]#cyst or lung
                        data = data[:,area[1].min():area[1].max(),area[2].min():area[2].max()]
                temporalvolume,pos,_ = self.bbc(data, self.padding,data_path,M)
            else:
                temporalvolume=self.get_all_jpgs(data_path,self.padding)
                pos=len(temporalvolume)
                #feature=np.zeros((self.padding,479))
            try:
                age = int(rawinfo['age'])
                gender = int(rawinfo['gender']=='M')
            except:
                age=int(rawinfo['age'][:-1])
                gender=int(rawinfo['gender']=='M')
        if self.options['general']['clinic']:
            feature=self.info[idx]
            feature=[feature]*pos
        else:
            feature=0
        
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'label2': torch.LongTensor([clsII]),
            'length':[data_path,pos],
            'gender': torch.LongTensor([gender]),
            'age': torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos]),
            'rfeature':torch.FloatTensor([rfeatures]),
            'cfeature':torch.FloatTensor([cfeatures])
            }
    def get_all_jpgs(self,data_path,padding):
        root="/mnt/data9/covid_detector_jpgs/byjb/"
        root2="/mnt/data9/covid_detector_jpgs/byjb_val/"
        type1=data_path.split('/')[4]
        type2=data_path.split('/')[5]
        name=data_path.split('/')[-1].split('-')[0]
        all_files=glob.glob(os.path.join(root,type1,type2,name+'_*.jpg'))+glob.glob(os.path.join(root2,type1,type2,name+'_*.jpg'))
        ids=[int(item.split('_')[-1].split('.')[0]) for item in all_files]
        
        id_s=np.argsort(ids)

        all_files=np.array(all_files)[id_s]
        all_files=all_files.tolist()
        temporalvolume = torch.zeros((3, padding, 224, 224))
        for idx,i in enumerate(all_files):
            data=Image.open(i)
            data = data.convert("RGB")
            data=self.transform(data)
            temporalvolume[:, idx] = data
        return temporalvolume

    def bbc(self,V, padding,data_path,pre=None,cc=None):
       
        F=np.zeros((padding,479))

        stride=max(V.shape[0]//padding,1)
        cnt=0
        name=[]
        if self.options['general']['mc']:
            temporalvolume = torch.zeros((15, padding, 224, 224))
        else:
            temporalvolume = torch.zeros((3, padding, 224, 224))
        for cnt,i in enumerate(range(0,V.shape[0],stride)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            data=V[i,:,:]
            data[data > 500] = 500
            data[data < -1200] = -1200

            name.append(float(i/V.shape[0]))
            data = data +1200
            data = data*255.0 /1700
            if self.options['general']['mc']:
                #temporalvolume = torch.zeros((15, padding, 224, 224))
                c=cc[i,:,:]
                m=pre[i, :, :]*255
                data = np.stack([data,data,data], -1)
                data = data.astype(np.uint8)
                data=Image.fromarray(data)
                data = data.convert("RGB")
                result = self.transform(data)
                temporalvolume[0, cnt,:,:] = result[0,:,:]
                m= np.stack([m,m,m], -1)
                m = m.astype(np.uint8)
                m=Image.fromarray(m)
                m = m.convert("RGB")
                result = self.transform(m)
                temporalvolume[1, cnt,:,:] = result[0,:,:]
                #cc=c[i,:,:]
                for j in range(1,14):
                    temp=(c==j)*255
                    temp= np.stack([temp,temp,temp], -1)
                    temp = temp.astype(np.uint8)
                    temp=Image.fromarray(temp)
                    temp = temp.convert("RGB")
                    result = self.transform(temp)
                    temporalvolume[j+1, cnt,:,:] = result[0,:,:]
            elif self.options['general']['stacked']:
                if i>=V.shape[0]-5:
                    break
                data=V[i:i+5,:,:].max(0)
                data[data > 500] = 500
                data[data < -1200] = -1200
                mask=pre[i:5+i,:,:].max(0)
                name.append(float(i/V.shape[0]))
                data = data +1200
                data = data*255.0 /1700
                data = np.stack([mask*255,mask * data,data], -1)  # mask one channel
                data = data.astype(np.uint8)
                data=Image.fromarray(data)
                data = data.convert("RGB")

                result = self.transform(data)
                temporalvolume[:, cnt] = result
            else: 
                #temporalvolume = torch.zeros((3, padding, 224, 224))
                data = np.stack([pre[i,:,:]*255,pre[i, :, :] * data,data], -1)  # mask one channel

                #data = np.stack([pre[i, :, :] * data, pre[i, :, :] * data, pre[i, :, :] *data], -1)
                data = data.astype(np.uint8)
                data=Image.fromarray(data)
                data = data.convert("RGB")
                #data = data.convert("RGB")
                #data.save('temp.jpg')
                #data.save('temp.jpg')
                result = self.transform(data)
                temporalvolume[:, cnt] = result

        #temporalvolume=temporalvolume
        return temporalvolume[:,:cnt,:,:],cnt,F[:cnt]


class DBZ_V_dataset(Dataset):
    def __init__(self, lists,padding,cls_num=6,options=None,logger=None,is_train=True):

        self.options = options
        self.cls_num=cls_num
        self.data = []
        self.mask=[]
        self.clsI=[]
        self.clsII = []
        self.cc=[]
        self.text_book=None
        self.mod='dbz'
        self.logger=logger
        self.infos=[]
        info=[]
        self.raw_data=json.load(open(lists,'r'))
        self.infofile=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/clinical_all.json','r'))
        for one in self.raw_data.keys():
            thisone=self.raw_data[one]
            clsI=maintype.index(thisone['clsI'])
            clsII=allsubtype.index(thisone['clsII'])
            if clsI in FILTERLIST1:
                if clsII  in FILTERLIST2:
                    if options['general']['clinic']:
                        id=thisone['newpath'].split('/')[-1].split('-')[0]
                        if id in self.infofile.keys():
                            info.append(self.infofile[id]['clinic_f'])
                        else:
                            continue
                    self.data.append(thisone['newpath'])
                    self.mask.append(thisone['newpath'].replace('data_for2d', 'seg_for2d'))
                    self.cc.append(thisone['newpath'].replace('newdisk3/data_for2d','newdisk2/reg/transform').replace('.nii','.seg.nii'))
                    self.clsI.append(clsI)
                    self.clsII.append(clsII)
                    self.infos.append(thisone)
        self.padding = padding
        self.logger.info('{} dataset'.format(lists))

        nums = [np.sum(np.array(self.clsI) == i) for i in range(np.max(self.clsI) + 1)]
        self.logger.info('maintype'+str(nums))
        nums = [np.sum(np.array(self.clsII) == i) for i in range(np.max(self.clsII) + 1)]
        self.logger.info('subtype'+str(nums))
        self.info=info
        self.is_train=is_train
        self.need_crop=True
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        rawinfo=self.infos[idx]
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        #direct=False
        #print(data_path,mask_path)
        cls=self.clsI[idx]
        clsII = self.clsII[idx]
        mask = sitk.ReadImage(mask_path)
        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        M=M[:data.shape[0],:,:]
        if M.max()==0:
            area = np.where(data >= 0)
        else:
            M[M>1]=1
            valid = np.where(M.sum(1).sum(1) > 1000)
            if len(valid) == 0:
                area = np.where(M >= 0)
            else:
                data = data[valid[0],:,:]
                M = M[valid[0],:,:]
                area = np.where(M>0)
        if self.need_crop:
            data=data[area[0].min():area[0].max(),area[1].min():area[1].max(),area[2].min():area[2].max()]
            M=M[area[0].min():area[0].max(),area[1].min():area[1].max(),area[2].min():area[2].max()]
        temporalvolume,cnt,_ = self.bbc(data, self.padding,augmentation=self.is_train,pre=M)
        try:
            age = int(rawinfo['age'])
            gender = int(rawinfo['gender']=='M')
        except:
            age=int(rawinfo['age'][:-1])
            gender=int(rawinfo['gender']=='M')
        feature=0
        pos=-1
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'label2': torch.LongTensor([clsII]),
            'length':[cnt],
            'gender': torch.LongTensor([gender]),
            'age': torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos]),
            'features':torch.FloatTensor(feature),
            'name':[data_path]
            }
    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weights for dealing with class imbalance problem

        """
        n_samples = len(self.clsI)
        unique, cnts = np.unique(self.clsI, return_counts=True)
        #cnts=cnts
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.clsI:
            weights.append((n_samples / float(cnt_dict[label])))

        return weights
    def bbc(self,V, padding,augmentation,pre=None,cc=None):
        F=np.zeros((padding,1))
        stride=max(V.shape[0]//padding,1)
        cnt=0
        name=[]
        temporalvolume = torch.zeros((3, padding, 224, 224))      
        if (augmentation):
            crop = StatefulRandomCrop((224, 224), (224, 224))
            flip = StatefulRandomHorizontalFlip(0.5)
            croptransform = transforms.Compose([
                crop,
                flip
            ])
        else:
            croptransform = transforms.CenterCrop((224, 224))
        for cnt,i in enumerate(range(0,V.shape[0],stride)):
            if cnt>=padding:
                break
            data=V[i,:,:]
            data[data > 500] = 500
            data[data < -1200] = -1200
            name.append(float(i/V.shape[0]))
            data = data +1200
            data = data*255.0 /1700
            data = np.stack([pre[i,:,:]*255,pre[i, :, :] * data,data], -1)  # mask one channel
            #data = np.stack([pre[i, :, :] * data, pre[i, :, :] * data, pre[i, :, :] *data], -1)
            data = data.astype(np.uint8)
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                #transforms.CenterCrop((256, 256)),
                croptransform,
                transforms.ToTensor(),
                transforms.Normalize([0, 0, 0], [1, 1, 1]),
            ])(data)
            temporalvolume[:, cnt] = result

        return temporalvolume,cnt,F


class Dmap_dataset(DBZ_dataset):
    def __init__(self, data_root,index_root, padding, augment=False,cls_num=6,mod='ab',options=None,logger=None,one_else=None):
        super(Dmap_dataset,self).__init__(data_root,index_root, padding, augment,cls_num,mod,options,logger,one_else)
    def __getitem__(self, idx):
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]      
        pmap_path=data_path.replace('.jpg','dmap.jpg')
        cls=self.labels[idx]
        cls2=self.labels2[idx] 
        rfeatures,cfeatures=0,0
        data=Image.open(data_path)
        pmap=Image.open(pmap_path)
        info=data_path.split('_')
        age,gender,pos=info[-3],info[-2],int(info[-1][:-4])
        try:
            age=int(age)
        except:
            age=int(age[:-1])
        gender=gender=='M'
        data = data.convert("RGB")
        pmap=pmap.convert('RGB')
        rotate=StatefulRotate([-15,15])
        crop = StatefulRandomCrop((224, 224), (224, 224))
        flip = StatefulRandomHorizontalFlip(0.5)
        croptransform = transforms.Compose([
            crop,
            flip,
            rotate
        ])
        if self.augment:
            data=transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                croptransform,
                transforms.ToTensor(),
                transforms.Normalize([0, 0, 0], [1, 1, 1]),
            ])(data)
            pmap=transforms.Compose([
                #transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                croptransform,
                transforms.ToTensor(),
                transforms.Normalize([0, 0, 0], [1, 1, 1]),
            ])(pmap)
        else:
            data=self.test_augmentation(data)
            pmap=self.test_augmentation(pmap)
        data=torch.cat([data,pmap],0)
        return {'temporalvolume': data,
            'label': torch.LongTensor([cls]),
            'length': torch.LongTensor([1]),
            'label2':torch.LongTensor([cls2]),
            'gender':torch.LongTensor([gender]),
            'age':torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos/100]),
            'name':data_path,
            'rfeature':torch.FloatTensor([rfeatures]),
            'cfeature':torch.FloatTensor([cfeatures])
            }


class Dmap_test_dataset(DBZ_test_dataset):
    def __init__(self, data_root,padding,lists,age_list=None,cls_num=6,mod='ab',options=None,logger=None,one_else=None):
        #self.padding = padding
        #self.data_root=data_root
        #if data_root[-3:]=='csv':
        #    self.r=pd.read_csv(data_root)
        self.filter='all'
        self.options = options
        self.rcdata=pickle.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/all_r2.pkl','rb'))
        self.cls_num=cls_num
        self.data = []
        self.mask=[]
        self.clsI=[]
        self.clsII = []
        self.cc=[]
        self.text_book=None
        self.mod='dbz'
        filter_list=open('data/clsuter.txt','r').readlines()
        severe_list=[a.split('\t')[0] for a in filter_list]
        id_list=[a.split('\t')[1][3:-5] for a in filter_list]
        self.logger=logger
        self.infos=[]
        info=[]
        self.raw_data=json.load(open(lists,'r'))
        self.infofile=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/clinical_all.json','r'))
        if one_else:
            level=int(one_else[0])
            ff=int(one_else[1:])
        for one in self.raw_data.keys():
            aa=1
            thisone=self.raw_data[one]
            pid=thisone['pid']
            if self.filter in ['mild','severe']:
                if not pid in id_list:
                    continue
                if not severe_list[id_list.index(pid)]==self.filter:
                    continue
            if not 'points_path' in thisone.keys():
                continue
            if not thisone['clsI'] in maintype:
                continue
            clsI=maintype.index(thisone['clsI'])
            try:
                clsII=allsubtype.index(thisone['clsII'])
            except:
                continue
            if clsI in FILTERLIST1:
                if clsII  in FILTERLIST2:
                    if options['general']['clinic']:
                        id=thisone['newpath'].split('/')[-1].split('-')[0]
                        if id in self.infofile.keys():
                            info.append(self.infofile[id]['clinic_f'])
                        else:
                            continue

                    self.data.append(thisone['newpath'])
                    self.mask.append(thisone['newpath'].replace('data_for2d', 'seg_for2d'))
                    self.cc.append(thisone['newpath'].replace('newdisk3/data_for2d','newdisk2/reg/transform').replace('.nii','.seg.nii'))

                    if one_else:
                        if level==1:
                            self.clsI.append(clsI==ff)
                            self.clsII.append(clsII)
                        else:
                            self.clsI.append(clsI)
                            self.clsII.append(clsII==ff)
                    else:
                        self.clsI.append(clsI)
                        self.clsII.append(clsII)
                    self.infos.append(thisone)
        self.padding = padding
        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         #transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        
        self.logger.info('{} dataset'.format(lists))
        nums = [np.sum(np.array(self.clsI) == i) for i in range(np.max(self.clsI) + 1)]
        self.logger.info('maintype'+str(nums))
        print('{} dataset'.format(lists))
        print('for maintype num is :'+str(nums))
        nums = [np.sum(np.array(self.clsII) == i) for i in range(np.max(self.clsII) + 1)]
        self.logger.info('subtype'+str(nums))
        print('for subtype num is :'+str(nums))
        self.info=info
    def __getitem__(self, idx):
        #load video into a tensor
        rawinfo=self.infos[idx]
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        #print(data_path,mask_path)
        cls=self.clsI[idx]
        clsII = self.clsII[idx]
        rfeatures,cfeatures=0,0
        mask = sitk.ReadImage(mask_path)
        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        
        dmap=np.load(rawinfo['points_path'])
        M=M[:data.shape[0],:,:]
        
        if M.max()==0:
            area = np.where(data >= 0)
        else:
            M[M>1]=1
            valid = np.where(M.sum(1).sum(1) > 500)
            if len(valid) == 0:
                area = np.where(M >= 0)
            else:
                data = data[valid[0],:,:]
                dmap=dmap[valid[0],:,:,:]
                M = M[valid[0],:,:]
                area = np.where(M>0)
                
        if self.options['input']['croped']:
                M = M[:,area[1].min():area[1].max(),area[2].min():area[2].max()]#cyst or lung
                data = data[:,area[1].min():area[1].max(),area[2].min():area[2].max()]
                dmap=dmap[:,area[1].min():area[1].max(),area[2].min():area[2].max(),:]
        temporalvolume,pos,_ = self.bbc(data, self.padding,M,dmap)
        
        try:
            age = int(rawinfo['age'])
            gender = int(rawinfo['gender']=='M')
        except:
            age=int(rawinfo['age'][:-1])
            gender=int(rawinfo['gender']=='M')

        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'label2': torch.LongTensor([clsII]),
            'length':[data_path,pos],
            'gender': torch.LongTensor([gender]),
            'age': torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos]),
            'rfeature':torch.FloatTensor([rfeatures]),
            'cfeature':torch.FloatTensor([cfeatures])
            } 
   
    def bbc(self,V, padding,pre=None,dmap=None):
       
        F=np.zeros((padding,479))
        stride=max(V.shape[0]//padding,1)
        cnt=0
        name=[]
        temporalvolume = torch.zeros((6, padding, 224, 224))
        for cnt,i in enumerate(range(0,V.shape[0],stride)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            data=V[i,:,:]
            pdata=dmap[i,:,:,:]
            pdata=pdata*np.tile(pre[i,:,:,np.newaxis],3)
            data[data > 500] = 500
            data[data < -1200] = -1200

            name.append(float(i/V.shape[0]))
            data = data +1200
            data = data*255.0 /1700

            data = np.stack([pre[i,:,:]*255,pre[i, :, :] * data,data], -1)  # mask one channel
            pdata=pdata[:,:,[2,1,0]]
            #data = np.stack([pre[i, :, :] * data, pre[i, :, :] * data, pre[i, :, :] *data], -1)
            data = data.astype(np.uint8)
            pdata= pdata.astype(np.uint8)
            data=Image.fromarray(data)
            pdata=Image.fromarray(pdata).convert("RGB")
            data = data.convert("RGB")
            #data = data.convert("RGB")
            #data.save('temp.jpg')
            #data.save('temp.jpg')
            result = self.transform(data)
            result2 = self.transform(pdata)
            temporalvolume[:3, cnt] = result
            temporalvolume[3:, cnt] = result2

        #temporalvolume=temporalvolume
        return temporalvolume[:,:cnt,:,:],cnt,F[:cnt]