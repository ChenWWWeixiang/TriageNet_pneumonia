#!/usr/bin/env python
# coding: utf-8


from baseline import CXR_dataset 

import time, shutil, os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, multilabel_confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

import torch
from torch import nn, optim
from torchvision import models, transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES']='2'

args = {
    'batch_size': 64, 
    'lr': 1e-4,
    'wd': 1e-5,
    'num_workers': 2,
    'num_epochs': 3
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[ ]:


# Function for randomly initializing weights.
def initialize_weights(net):

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
            
    return net

def adapt_net(net, name, num_classes):
    
    if 'vgg' in name:
        initial_layer = net.features[0]
        net.features = nn.Sequential(nn.Conv2d(1, initial_layer.out_channels, initial_layer.kernel_size,
                                               stride=initial_layer.stride, padding=initial_layer.padding),
                                    *list(net.features.children())[1:])
        
        in_features = net.classifier[-1].in_features
        new_classifier = nn.Sequential(*list(net.classifier.children())[:-1],
                                       nn.Linear(in_features, num_classes),
                                       nn.Sigmoid())
        net.classifier = new_classifier
    
    elif 'resnet' in name or 'resnext' in name:
        initial_layer = net.conv1
        net.conv1 = nn.Conv2d(1, initial_layer.out_channels, initial_layer.kernel_size,
                              stride=initial_layer.stride, padding=initial_layer.padding)
        
        in_features = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(in_features, num_classes),
                               nn.Sigmoid())
    
    elif 'densenet' in name:
        initial_layer = net.features[0]
        net.features = nn.Sequential(nn.Conv2d(1, initial_layer.out_channels, initial_layer.kernel_size,
                                               stride=initial_layer.stride, padding=initial_layer.padding),
                                    *list(net.features.children())[1:])
        
        in_features = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Linear(in_features, num_classes),
                                       nn.Sigmoid())
               
    net = initialize_weights(net)
    return net


def forward(epoch, loader, net, criterion, optimizer, tensorboard, classes, mode, thres=0.1):
    
    start = time.time()
    start_epoch = start 

    plot_every = len(loader)
    #plot_every=1
    

    all_pred, all_label = [], []
    losstensorMean = 0
    
    for i, (img, label,name) in enumerate(loader):

        img = img.to(device).float()
        label = label.to(device).float()
        label[label==-1]=0
        output = net(img)
        loss = criterion(output, label.detach())
        
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losstensorMean += loss.detach()
        all_pred.append(output.detach().cpu().numpy())
        all_label.append(label.detach().cpu().numpy())
        del img, label, output, loss
        torch.cuda.empty_cache()
        gc.collect()
        
    
    print(f'\r{i}/{len(loader)} ----{time.time()-start:.2f}----', end='', flush=True)
    
    losstensorMean = losstensorMean/plot_every
        
    tensorboard.add_scalar(f'Loss/{mode}',
                            losstensorMean,
                            epoch * len(loader) + i)
    
    all_pred=np.concatenate(all_pred,0)
    all_label=np.concatenate(all_label,0)
    auc_mean = 0.
    for k in range(len(classes)):
        
        roc_auc = roc_auc_score(all_label[:, k], all_pred[:, k])
        auc_mean += roc_auc
        
        tensorboard.add_scalar(f'AUC_{mode}/{classes[k]}',
                                roc_auc,
                                epoch * len(loader) + i)
        print(f'AUC_{mode}/{classes[k]}',roc_auc,epoch * len(loader) + i)
    tensorboard.add_scalar(f'AUC_{mode}/Mean',
                                auc_mean/len(classes),
                                epoch * len(loader) + i)
    print(f'AUC_{mode}/Mean',auc_mean/len(classes),epoch * len(loader) + i)
    start = time.time()

    del roc_auc, auc_mean 
    #del all_pred_, all_label_
    torch.cuda.empty_cache()
    gc.collect()
    print(f'\n[{mode.capitalize()}] Epoch: {epoch+1} - Loss: {losstensorMean:.4f} - Time {time.time()-start_epoch}')
    return losstensorMean


dataloaders_dict = {}

valid_data_list=open('/mnt/newdisk3/CheXpert-v1.0-small/valid.csv','r').readlines()
train_data_list=open('/mnt/newdisk3/CheXpert-v1.0-small/train.csv','r').readlines()
records=train_data_list[0].split(',')[-14:]

normalize = transforms.Normalize(mean=[0.497], std=[0.087])

data_train = CXR_dataset('/mnt/newdisk3',train_data_list,True, t=transforms.Compose([
                                                                transforms.ToPILImage(),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomRotation(10),
                                                                transforms.Scale(224),
                                                                transforms.CenterCrop(224),
                                                                transforms.ToTensor(),
                                                                normalize
                                                            ]))

data_test = CXR_dataset('/mnt/newdisk3',valid_data_list,True, t=transforms.Compose([
                                                            transforms.ToPILImage(),
                                                            transforms.Scale(224),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            normalize
                                                        ]))

dataloaders_dict['train'] = torch.utils.data.DataLoader(data_train,
    batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

dataloaders_dict['val'] = torch.utils.data.DataLoader(data_test,
    batch_size=args['batch_size'], shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

classes = np.array(train_data_list[0].split(','))[np.arange(-14,-2).tolist()+[-1]]


if not os.path.isdir('models'):
    os.mkdir('models')


architec = [
            'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
            'resnet18', 'resnet34', 'resnet50', 'resnet152',
            'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'resnext50_32x4d', 'wide_resnet50_2', 'resnext101_32x8d', 'wide_resnet101_2'
]


######################
pretrained = False
dataset = 'CheXpert'
######################

for arch in architec:
    print('\n','*'*30, arch, '*'*30)

    writer_path = f'runs/{dataset}_{arch}_{pretrained}'
    model_path = f'models/{dataset}_{arch}_{pretrained}'

    writer = SummaryWriter(writer_path)

    model = getattr(models, arch)
    net = model(pretrained=pretrained)
    net = adapt_net(net, arch, len(classes))
    net = net.to(device)
    
    criterion = torch.nn.BCELoss(reduction='mean').to(device)
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], betas=(0.9, 0.999), eps=1e-08, 
                           weight_decay=args['wd'], amsgrad=True)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    min_val_loss = np.inf
    
    for epoch in range(args['num_epochs']):

        losstensor = forward(epoch, dataloaders_dict['train'], net, criterion, optimizer, writer, classes, 'train')
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            losstensor = forward(epoch, dataloaders_dict['val'],  net, criterion, optimizer, writer, classes, 'valid')
            scheduler.step(losstensor)
        
            if losstensor < min_val_loss:
                min_val_loss = losstensor
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print(f'Model saved.')
            
        torch.cuda.empty_cache()
        gc.collect()
        print('-'*50) 
        
    del net, criterion, optimizer, writer
    torch.cuda.empty_cache()
    gc.collect()

