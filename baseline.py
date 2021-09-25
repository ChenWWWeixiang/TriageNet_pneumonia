import cv2,csv,os,torch,random
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms.transforms import RandomRotation
from models.net2d import model_urls,input_sizes,means,stds,load_pretrained,ForestCLF
import os
from sklearn.metrics import auc,roc_curve
import sklearn.metrics as metric
os.environ['CUDA_VISIBLE_DEVICES']='1'

class CXR_dataset(Dataset):
    def __init__(self, index_root,data_list,istrain,t=None):
        self.istrain=istrain
        self.data_list=data_list
        split_items=np.array([item.split(',') for item in self.data_list])
        self.labels=split_items[1:,np.arange(-14,-2).tolist()+[-1]]
        self.labels[self.labels=='']='-1'
        self.labels[self.labels=='\n']='-1'
        self.labels=self.labels.astype(np.float)
        #self.labels=self.labels[]
        self.data_path=[os.path.join(index_root,id) for id in split_items[1:,0]]
        print('num of data:', len(self.data_list))
        if t:
            self.transform=t
            self.test_transform=t
        else:   
            self.transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((320, 320)),
                    transforms.RandomCrop((320, 320)),
                    transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize([0, 0, 0], [1, 1, 1])])
            self.test_transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((320, 320)),
                    transforms.CenterCrop((320, 320)),
                    transforms.ToTensor(),
                    transforms.Normalize([0, 0, 0], [1, 1, 1])])

    def __len__(self):
        return len(self.data_path)
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
        I=cv2.imread(self.data_path[idx])[:,:,0]
        label=self.labels[idx,:]
        if self.istrain:
            I=self.transform(I)
        else:
            I=self.test_transform(I)
        return I,label,self.data_path[idx]

    def do_augmentation(self, array, mask):
       pass
def main():
    used_list=[2,5,6,8,10]
    valid_data_list=open('/mnt/newdisk3/CheXpert-v1.0-small/valid.csv','r').readlines()
    train_data_list=open('/mnt/newdisk3/CheXpert-v1.0-small/train.csv','r').readlines()
    records=train_data_list[0].split(',')[-14:]
    
    train_dataset=CXR_dataset('/mnt/newdisk3',train_data_list,True)
    trainingdataloader = DataLoader(train_dataset,
                                    batch_size=64,
                                    shuffle=True,
                                    num_workers=8,
                                    drop_last=False,
                                    #sampler=sampler,
                                    pin_memory=True,
                                    )
    test_dataset=CXR_dataset('/mnt/newdisk3',valid_data_list,False)
    testingdataloader = DataLoader(test_dataset,
                                    batch_size=64,
                                    shuffle=False,
                                    num_workers=8,
                                    drop_last=False,
                                    #sampler=sampler,
                                    pin_memory=True,
                                    )
    class myNet(nn.Module):
        def __init__(self):
            super().__init__()
            model=models.densenet121(pretrained=True).cuda()
            self.features=model.features
            del model.classifier
            self.classifier=nn.Linear(1024,5).cuda()
        def forward(self,x):
            features=self.features(x)
            features = F.relu(features, inplace=True)
            w,h=features.shape[-2],features.shape[-1]
            out = F.adaptive_avg_pool2d(features, (1, 1))#n,c,1,1
            out = torch.flatten(out, 1)
            #out=out.repeat(1,1,w,h)
            #features_c=torch.cat([features,out],1)
            pred=self.classifier(out)
            return pred


    model=myNet().cuda()
    optimizer=optim.Adam(model.parameters(),lr = 0.0001,amsgrad=True,betas=[0.9,0.999])
    cri=torch.nn.BCEWithLogitsLoss(reduction='none')
    ovefall_auc_list=[]
    for epoch in range(4):
        for idx,(item,label,_) in enumerate(trainingdataloader):
            #break
            optimizer.zero_grad()
            item,label=item.cuda().float(),label.cuda().float()
            label=label[:,np.array(used_list)]
            output=model(item)

            loss=F.binary_cross_entropy(output.sigmoid(),label,reduce=False)
            loss=loss[label>-1].mean()
            loss.backward()
            optimizer.step()
            if idx%10==1:
                print(idx,loss)
                #break
            if idx%1000==1:
                Acc=[]
                Acci=np.zeros(5)
                count=np.zeros(5)
                pred_list=[]
                label_list=[]
                result_dict=dict()
                gt_dict=dict()
                for idx,(item,label,path) in enumerate(testingdataloader):
                    item,label=item.cuda().float(),label.cuda().float()
                    label=label[:,np.array(used_list)]
                    names=[p.split('/')[-3] for p in path]
                    em=model(item).sigmoid().detach().cpu().numpy()
                    label=label.cpu().numpy()
                    for item in range(len(names)):
                        if names[item] in result_dict.keys():
                            result_dict[names[item]].append(em[item,:])
                            gt_dict[names[item]].append(label[item,:])
                        else:
                            result_dict[names[item]]=[em[item,:]]
                            gt_dict[names[item]]=[label[item,:]]
                for item in gt_dict.keys():
                    #print(np.mean(gt_dict[item],0).shape)
                    label_list.append(np.mean(gt_dict[item],0))
                    pred_list.append(np.mean(result_dict[item],0))
                    
                #Acc=np.mean(Acc)
                pred_list=np.array(pred_list)
                label_list=np.array(label_list)
                ovefall_auc = metric.roc_auc_score(label_list, pred_list, average='micro')
                ovefall_auc_list.append([idx,ovefall_auc])
                os.makedirs('saved_chexpert',exist_ok=True)
                torch.save(model.state_dict(), f"saved_chexpert/{idx}_baseline_model.pt")
        ovefall_auc_list_name=[item[0] for item in ovefall_auc_list]
        ovefall_auc_list_val=[item[1] for item in ovefall_auc_list]
        ovefall_auc_list_val=np.array(ovefall_auc_list_val)
        selected_model=np.argsort(-ovefall_auc_list_val)[:10]
        selected_model=[f"saved_chexpert/{ovefall_auc_list_name[i]}_baseline_model.pt" for i in selected_model]
        AUCs=[[]]*5
        for one in selected_model:
            model.load_state_dict(torch.load(one))
            result_dict=dict()
            gt_dict=dict()
            for idx,(item,label,path) in enumerate(testingdataloader):
                    item,label=item.cuda().float(),label.cuda().float()
                    label=label[:,np.array(used_list)]
                    names=[p.split('/')[-3] for p in path]
                    em=model(item).sigmoid().detach().cpu().numpy()
                    logits=(em>0.5)
                    label=label.cpu().numpy()
                    acc=(label==logits)*1.0
                    acc=acc[label>-1].mean()
                    for item in range(len(names)):
                        if names[item] in result_dict.keys():
                            result_dict[names[item]].append(em[item,:])
                            gt_dict[names[item]].append(label[item,:])
                        else:
                            result_dict[names[item]]=[em[item,:]]
                            gt_dict[names[item]]=[label[item,:]]
            for i in range(5):
                this_pred=pred_list[label_list[:,i]>-1,i]
                this_label=label_list[label_list[:,i]>-1,i]
                try:
                    this_auc = metric.roc_auc_score(this_label, this_pred, average='micro')
                except:
                    this_auc=0
                AUCs[i].append(this_auc)
        AUCs=np.stack(AUCs,1).mean(1)
        records=np.array(records)[np.array(used_list)]
        with open('baseline_v2.csv','a') as f:
            for i in range(5):
                print(records[i],'auc',AUCs[i])
                f.writelines(records[i]+',auc,'+str(AUCs[i])+'\n')

main()