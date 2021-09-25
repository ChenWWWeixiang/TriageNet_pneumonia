import cv2,csv,os,torch,random
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms.transforms import RandomRotation
from models.net2d import model_urls,input_sizes,means,stds,load_pretrained,ForestCLF
import os
import sklearn.metrics as metric
from sklearn.metrics import auc,roc_curve
os.environ['CUDA_VISIBLE_DEVICES']='7'
    
used_list=np.array([2,5,6,8,10])
class ForestCLF_chexpert(ForestCLF):
    def __init__(self,num_trees,model,num_features=1024,num_of_cls=3,num_of_cls2=3):
        super(ForestCLF,self).__init__()
        self.cls_agents=[]
        self.num_of_cls2=num_of_cls2
        self.num_trees=num_trees
        self.num_features=num_features
        self.num_of_cls=num_of_cls

        self.backbone=model.features

        self.cc=[]
        self.idx=[]
        self.clc=[]
        self.cls_agents=nn.ModuleList()
        for i in range(self.num_trees):
            self.cls_agents.append(nn.Linear(num_features,num_of_cls2).cuda())    
            self.clc.append(nn.Linear(self.num_features,2).cuda())
        self.lossfunction=torch.nn.BCEWithLogitsLoss(reduction='none')

    def init_params(self):
        f_lists=np.arange(0,1024)
        c_lists2=np.arange(0,14)
        for i in range(self.num_trees):
            self.idx.append(np.array(random.sample(f_lists.tolist(), self.num_features)))
            self.cc.append(np.array(random.sample(c_lists2.tolist(), self.num_of_cls2)))
        manually_cc=[[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,0]]
        self.cc[:5]=[np.array(one) for one in manually_cc]
    def get_output(self,x):
        d=self.backbone(x)
        d = F.relu(d, inplace=True)
        d = F.adaptive_avg_pool2d(d, (1, 1))
        d = torch.flatten(d, 1)

        valid=torch.where((~torch.isnan(d)).any(1))
        d=d[valid]
        result=[]
        ws=[]
        for i in range(self.num_trees):
            w=self.clc[i](d[:,self.idx[i]]).log_softmax(-1)
            y=self.cls_agents[i](d[:,self.idx[i]])
            result.append(y)
            ws.append(w)
            
        output_numpy=np.zeros((result[0].shape[0],14))
        count=np.zeros((result[0].shape[0],14))
        #output_numpy2_=np.zeros_like(output_numpy2)
        for i in range(self.num_trees):
            output_numpy[:,self.cc[i]]+=result[i].sigmoid().detach().cpu().numpy()*torch.exp(ws[i][:,1:2]).detach().cpu().numpy()
            count[:,self.cc[i]]+=torch.exp(ws[i][:,1:2]).detach().cpu().numpy()
        output_numpy=output_numpy/(count)
        output_numpy[np.isnan(output_numpy)]=0
        return output_numpy
    
    def get_pre1(self,gt):
        output_numpy=np.zeros((self.result[0].shape[0],14))
        count=np.zeros((self.result[0].shape[0],14))
        for i in range(self.num_trees):
            output_numpy[:,self.cc[i]]+=torch.exp(self.result[i].sigmoid()).detach().cpu().numpy()*torch.exp(self.w[i][:,1:2]).detach().cpu().numpy()
            count[:,self.cc[i]]+=torch.exp(self.w[i][:,1:2]).detach().cpu().numpy()
        output_numpy=output_numpy/(count)  
        output_numpy[np.isnan(output_numpy)]=0
        output_numpy=output_numpy/output_numpy.sum(-1,keepdims=True)
        pre=np.argmax(output_numpy,-1)
        return pre
    
    def forward(self,x,target):
        d=self.backbone(x)
        d = F.relu(d, inplace=True)
        d = F.adaptive_avg_pool2d(d, (1, 1))
        d = torch.flatten(d, 1)
        valid=torch.where((~torch.isnan(d)).any(1))
        d=d[valid]
        target=target[valid]
        #target2=target2[valid]  
        self.result=[]
        loss2=[]
        loss3=[]
        self.w=[]
        for i in range(self.num_trees):
            w=self.clc[i](d[:,self.idx[i]]).log_softmax(-1)#      
            self.w.append(w)
            y=self.cls_agents[i](d[:,self.idx[i]])
            self.result.append(y)
            target_=target[:,self.cc[i]].float()
            idx_used=(target_>=0).any(1)
            if idx_used.sum()>0:
                lx=F.binary_cross_entropy(y[idx_used,:].sigmoid(),target_[idx_used,:],reduce=False)
                lx=lx[target_[idx_used,:]>-1].mean()
                factor=len(idx_used)/idx_used.sum()/2
                loss2.append(lx*factor)
            w_gt=idx_used.long()
            loss3.append(F.nll_loss(w,w_gt))
        loss2=torch.stack(loss2)
        loss3=torch.stack(loss3)
        loss2=torch.mean(loss2)
        loss3=torch.mean(loss3)
        #wl=loss3.detach()
        loss=loss2+loss3
        return loss,loss2,loss3

class ForestCLF_chexpert_v2(ForestCLF):
    def __init__(self,num_trees,model,num_features=1024,num_of_cls=3,):
        super(ForestCLF,self).__init__()
        #self.cls_agents=[]
        self.num_of_cls2=num_of_cls
        self.num_trees=num_trees
        self.num_features=num_features
        self.num_of_cls=num_of_cls
        self.backbone=model.features
        self.cc=[]
        self.idx=[]
        self.clc=[]
        self.cls_agents=nn.ModuleList()

        self.clc=nn.Linear(1024,num_trees).cuda()
        #self.lossfunction=torch.nn.BCEWithLogitsLoss(reduction='none')s
    def fixed_groups_params(self):
        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([0,1,2,3,4]))

        self.idx.append(np.arange(0,1024))
        self.cc.append(np.array([1,2]))

        self.idx.append(np.arange(0,1024))
        self.cc.append(np.array([0,2]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([5,2]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([2,5]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([5]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([2]))

        self.idx.append(np.arange(0,1024))
        self.cc.append(np.array([2,3]))

        self.idx.append(np.arange(0,1024))
        self.cc.append(np.array([3,4,0]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([0,1,2]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([2,3,4]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([5,6,7,8,9]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([5,10,11]))
        
        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([5,10,12,13]))

        # self.idx.append(np.arange(0,1024))
        # self.cc.append(np.array([5,2,0,11]))

        self.num_trees=100
        f_lists=np.arange(0,1024)
        c_lists2=np.arange(0,len(used_list))
        for i in range(len(self.cc)):
            self.cls_agents.append(nn.Linear(1024,len(self.cc[i])).cuda())

        for i in range(len(self.idx),self.num_trees):
            self.idx.append(np.array(random.sample(f_lists.tolist(), self.num_features)))
            self.cc.append(np.array(random.sample(c_lists2.tolist(), self.num_of_cls2)))
            self.cls_agents.append(nn.Linear(self.num_features,self.num_of_cls2).cuda())   
 

    def init_params(self):
        f_lists=np.arange(0,1024)
        c_lists2=np.arange(0,14)
        for i in range(self.num_trees):
            self.idx.append(np.array(random.sample(f_lists.tolist(), self.num_features)))
            self.cc.append(np.array(random.sample(c_lists2.tolist(), self.num_of_cls2)))
            self.cls_agents.append(nn.Linear(self.num_features,self.num_of_cls2).cuda())    

    def get_output(self,x):
        d=self.backbone(x)
        d = F.relu(d, inplace=True)
        d = F.adaptive_avg_pool2d(d, (1, 1))
        d = torch.flatten(d, 1)

        valid=torch.where((~torch.isnan(d)).any(1))
        d=d[valid]
        result=[]
        ws=[]
        ws=self.clc(d).sigmoid()
        for i in range(self.num_trees):
            y=self.cls_agents[i](d[:,self.idx[i]])
            result.append(y)
            
        output_numpy=np.zeros((result[0].shape[0],5))
        count=np.zeros((result[0].shape[0],5))
        #output_numpy2_=np.zeros_like(output_numpy2)
        for i in range(self.num_trees):
            output_numpy[:,self.cc[i]]+=result[i].sigmoid().detach().cpu().numpy()*ws[:,i:i+1].detach().cpu().numpy()
            count[:,self.cc[i]]+=ws[:,i:i+1].detach().cpu().numpy()
        output_numpy=output_numpy/(count)
        output_numpy[np.isnan(output_numpy)]=0
        return output_numpy
    
    def forward(self,x,target):
        d=self.backbone(x)
        d = F.relu(d, inplace=True)
        d = F.adaptive_avg_pool2d(d, (1, 1))
        d = torch.flatten(d, 1)
        valid=torch.where((~torch.isnan(d)).any(1))
        d=d[valid]
        target=target[valid]
        #target2=target2[valid]  
        self.result=[]
        loss2=[]
        loss3=[]
        self.w=self.clc(d).sigmoid()
        gt_w=[]
        for i in range(self.num_trees):
            
            y=self.cls_agents[i](d[:,self.idx[i]])
            self.result.append(y)
            target_=target[:,self.cc[i]].float()
            idx_used=(target_>=0).any(1)
            if idx_used.sum()>0:
                lx=F.binary_cross_entropy(y[idx_used,:].sigmoid(),target_[idx_used,:],reduce=False)
                lx=lx[target_[idx_used,:]>-1].mean()
                factor=len(idx_used)/idx_used.sum()/2
                loss2.append(lx*factor)
            w_gt=idx_used.float()
            gt_w.append(w_gt)
        gt_w=torch.stack(gt_w,-1)
        loss3=F.binary_cross_entropy(self.w,gt_w)
        loss2=torch.stack(loss2)
        loss2=torch.mean(loss2)
        #wl=loss3.detach()
        loss=loss2+loss3*5
        return loss,loss2,loss3


class CXR_dataset(Dataset):
    def __init__(self, index_root,data_list,istrain):
        self.istrain=istrain
        self.data_list=data_list
        split_items=np.array([item.split(',') for item in self.data_list])
        self.labels=split_items[1:,-14:]
        self.labels=self.labels[:,used_list]
        self.labels[self.labels=='']='-1'
        self.labels[self.labels=='\n']='-1'
        self.labels=self.labels.astype(np.float)
        self.data_path=[os.path.join(index_root,id) for id in split_items[1:,0]]
        print('num of data:', len(self.data_list))
        self.transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((355, 355)),
                    transforms.RandomCrop((320, 320)),
                    transforms.RandomRotation(10),
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
        I=cv2.imread(self.data_path[idx])
        label=self.labels[idx,:]
        if self.istrain:
            I=self.transform(I)
        else:
            I=self.test_transform(I)
        return I,label,self.data_path[idx]

    def do_augmentation(self, array, mask):
       pass

valid_data_list=open('/mnt/newdisk3/CheXpert-v1.0-small/valid.csv','r').readlines()
train_data_list=open('/mnt/newdisk3/CheXpert-v1.0-small/train.csv','r').readlines()
records=train_data_list[0].split(',')[-14:]
records=np.array(records)[used_list].tolist()

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
def backbone_define():
    model= models.resnet152(pretrained=False)
    settings = pretrained_settings['resnet152']['imagenet']
    model = load_pretrained(model, 1000, settings)
    del model.fc
    model.fc=torch.nn.Linear(2048,14)
    model=model.cuda()
    return model

model=models.densenet121(pretrained=True).cuda()

model=ForestCLF_chexpert_v2(100,model,860,2)
model.fixed_groups_params()
optimizer=optim.Adam(model.parameters(),lr = 0.0001,amsgrad=True,betas=[0.9,0.999])
cri=torch.nn.BCEWithLogitsLoss(reduction='none')
ovefall_auc_list=[]

for epoch in range(5):
    for idx,(item,label,_) in enumerate(trainingdataloader):
        #break
        optimizer.zero_grad()
        item,label=item.cuda(),label.cuda()
        loss,l1,l2=model(item,label)
        loss.backward()
        optimizer.step()
        if idx%10==1:
            print(idx,loss.item(),l1.item(),l2.item())
        if idx%1000==1:
            Acc=[]
            pred_list=[]
            label_list=[]
            result_dict=dict()
            gt_dict=dict()
            for idx,(item,label,path) in enumerate(testingdataloader):
                item,label=item.cuda().float(),label.cuda().float()
                names=[p.split('/')[-3] for p in path]
                em=model.get_output(item)
                label=label.cpu().numpy()
                for item in range(len(names)):
                    if names[item] in result_dict.keys():
                        result_dict[names[item]].append(em[item,:])
                        gt_dict[names[item]].append(label[item,:])
                    else:
                        result_dict[names[item]]=[em[item,:]]
                        gt_dict[names[item]]=[label[item,:]]
            for item in gt_dict.keys():
                label_list.append(np.mean(gt_dict[item],0))
                pred_list.append(np.mean(result_dict[item],0))
                
            #Acc=np.mean(Acc)
            pred_list=np.array(pred_list)
            label_list=np.array(label_list)
            ovefall_auc = metric.roc_auc_score(label_list, pred_list, average='micro')
            ovefall_auc_list.append([idx,ovefall_auc])
            os.makedirs('saved_chexpert',exist_ok=True)
            torch.save(model.state_dict(), f"saved_chexpert/{idx}_triagenet_model.pt")
        
    ovefall_auc_list_name=[item[0] for item in ovefall_auc_list]
    ovefall_auc_list_val=[item[1] for item in ovefall_auc_list]
    ovefall_auc_list_val=np.array(ovefall_auc_list_val)
    selected_model=np.argsort(-ovefall_auc_list_val)[:10]
    selected_model=[f"saved_chexpert/{ovefall_auc_list_name[i]}_triagenet_model.pt" for i in selected_model]
    AUCs=[[]]*len(used_list)
    for one in selected_model:
        model.load_state_dict(torch.load(one))
        result_dict=dict()
        gt_dict=dict()
        for idx,(item,label,path) in enumerate(testingdataloader):
                item,label=item.cuda().float(),label.cuda().float()
                names=[p.split('/')[-3] for p in path]
                em=model.get_output(item)
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
        for i in range(len(used_list)):
            this_pred=pred_list[label_list[:,i]>-1,i]
            this_label=label_list[label_list[:,i]>-1,i]
            try:
                this_auc = metric.roc_auc_score(this_label, this_pred, average='micro')
            except:
                this_auc=0
            AUCs[i].append(this_auc)
    AUCs=np.stack(AUCs,1).mean(1)
    with open('triagenet_v2.csv','a') as f:
        for i in range(len(used_list)):
            print(records[i],'auc',AUCs[i])
            f.writelines(records[i]+',auc,'+str(AUCs[i])+'\n')


        
        