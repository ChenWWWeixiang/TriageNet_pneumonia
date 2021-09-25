from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from datetime import datetime, timedelta

from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import Dmap_dataset,NCPDataset,NCP2DDataset,NCPJPGDataset,NCPJPGDataset_new,NCPJPGtestDataset_new,DBZ_dataset,DBZ_V_dataset
import os,cv2
import numpy as np
from for_dbz_pre import settings2d
TYPEMAT=[[0,1,2,3,4],[5,6,7,8],[9,10,11,12,13],[14],[15]]
def _validate(modelOutput, length, labels, total=None, wrong=None):

    averageEnergies = torch.mean(modelOutput.data, 1)
    for i in range(modelOutput.size(0)):
        #print(modelOutput[i,:length[i]].sum(0).shape)
        averageEnergies[i] = modelOutput[i,:length[i]].mean(0)

    maxvalues, maxindices = torch.max(averageEnergies, 1)
    #print(maxindices.cpu().numpy())
    #print(labels.cpu().numpy())
    count = 0

    for i in range(0, labels.squeeze(1).size(0)):
        l = int(labels.squeeze(1)[i].cpu())
        if total is not None:
            if l not in total:
                total[l] = 1
            else:
                total[l] += 1
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
        else:
            if wrong is not None:
               if l not in wrong:
                   wrong[l] = 1
               else:
                   wrong[l] += 1

    return (averageEnergies, count)

class Validator():
    def __init__(self, options, mode,model,savenpy,logger):
        self.logger=logger
        self.options=options
        self.USE_SUB=options['model']['subcls']
        self.R=options['general']['clinic']
        self.model=model
        self.cls_num=options['general']['class_num']
        self.use_plus = options['general']['use_plus']
        self.use_3d=options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice=options['general']['use_slice']
        self.asinput = options['general']['plus_as_input']
        self.USE_25D = options['general']['use25d']
        mod=options['general']['mod']
        self.mod=mod
        if options['general']['use_slice']:
            if self.USE_25D:
                self.validationdataset = DBZ_V_dataset(options["validation"]["index_root"],options["training"]["padding"],
                    cls_num=self.cls_num,options=options,logger=self.logger,is_train=False)
            else:
                if mod=='dbz' or mod=='reader2'  or mod=='reader1':
                    self.validationdataset=DBZ_dataset(options[mode]["data_root"],
                                                           options['validation']['index_root'],
                                                           options[mode]["padding"],
                                                           augment=False,cls_num=self.cls_num,mod=options['general']['mod'],logger=self.logger,
                                                           options=options,one_else=options['one_else'])
                    # self.validationdataset=Dmap_dataset(options[mode]["data_root"],
                    #                                        options['validation']['index_root'],
                    #                                        options[mode]["padding"],
                    #                                        augment=False,cls_num=self.cls_num,mod=options['general']['mod'],logger=self.logger,
                    #                                        options=options,one_else=options['one_else'])
                else:
                    self.validationdataset = NCPJPGDataset_new(options[mode]["data_root"],
                                                        options[mode]["index_root"],
                                                        options[mode]["padding"],
                                                        False,cls_num=self.cls_num,
                                                        mod=options['general']['mod'],
                                                           options=options)
        else:
            if options['general']['use_3d']:
                self.validationdataset = NCPDataset(options[mode]["index_root"],
                                                      options[mode]["padding"],
                                                      False,
                                                    z_length=options["model"]["z_length"])
            else:
                self.validationdataset = NCP2DDataset(options[mode]["data_root"],
                                                        options[mode]["index_root"],
                                                        options[mode]["padding"],
                                                        False)
        self.savingnpy=savenpy

        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=True,
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False
                                )
        self.mode = mode
        self.epoch=0
        
    def __call__(self,writer):
        self.epoch+=1
        C=[]
        F=[]
        with torch.no_grad():
            self.logger.info("Starting {}...".format(self.mode))
            count = np.zeros((self.cls_num+self.use_plus*2*(1-self.asinput)))
            Matrix=np.zeros((self.cls_num,self.cls_num))
            num_samples2 = np.zeros((len(settings2d.allsubtype) + self.use_plus * 2*(1-self.asinput)))
            count2 = np.zeros((len(settings2d.allsubtype)))
            Matrix2 = np.zeros((len(settings2d.allsubtype), len(settings2d.allsubtype)))
            if self.use_3d:
                validator_function = self.model.validator_function()
            if self.use_lstm:
                validator_function = _validate
                self.model.eval()
            LL=[]
            GG=[]
            AA=[]
            #if(self.usecudnn):
            net = nn.DataParallel(self.model).cuda()
            error_dir='error/'
            os.makedirs(error_dir,exist_ok=True)
            cnt=0
            num_samples = np.zeros((self.cls_num+self.use_plus*2*(1-self.asinput)))
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                labels2 = Variable(sample_batched['label2']).cuda()
                if self.options['use_rc']:
                    r_feature = Variable(sample_batched['rfeature']).cuda()
                    c_feature = Variable(sample_batched['cfeature']).cuda()
                #length = len(sample_batched['length'][1])
                names=sample_batched['name'][0]
                if self.options['model']['forest']:
                    if self.options['use_rc']:
                        loss,loss1,loss2,loss3,deep_feature= net(input,r_feature,c_feature,labels,labels2)
                    else:
                        loss,loss1,loss2,loss3,deep_feature= net(input,labels,labels2)
                    pre,pre2=net.module.get_pre1(labels)
                   
                else:
                    if self.USE_SUB:
                        if self.R:
                            outputs,deep_feature,output2 = net(input,features,False)##here
                        else:
                            outputs,deep_feature,output2 = net(input,False)##here
                    else:
                        outputs,f = net(input)
                    F.append(deep_feature)
                    C.append(labels)
                    output_numpy = np.exp(outputs.cpu().numpy())
                    output_numpy2=np.zeros_like(output2.cpu().numpy())

                    output_numpy2= np.exp(output2.cpu().numpy())

                    
                    if self.USE_25D:
                        output_numpy=output_numpy.mean(1)
                        output_numpy2=output_numpy2.mean(1)
                    #output_numpy=output_numpy[:,[0,-1]]
                    #output_numpy_ab=np.sum(output_numpy[:,1:],1)
                    #output_numpy=np.stack([output_numpy[:,0],output_numpy_ab],-1)
                    output_numpy=output_numpy/output_numpy.sum(-1,keepdims=True)
                    output_numpy2=output_numpy2/output_numpy2.sum(-1,keepdims=True)
                    pre=np.argmax(output_numpy,-1)
                    pre2=np.argmax(output_numpy2,-1)
                isacc=labels.cpu().numpy().reshape(labels.size(0))==pre
                label_numpy=labels.cpu().numpy()[:,0]
                isacc2=labels2.cpu().numpy().reshape(labels2.size(0))==pre2
                label_numpy2=labels2.cpu().numpy()[:,0]
                #if self.tot_iter%10==0:

               # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    #LL.append([names[i], output_numpy[i,:], label_numpy[i]])
                    Matrix[label_numpy[i],pre[i]]+=1
                    try:
                        Matrix2[label_numpy2[i],pre2[i]]+=1
                    except:
                        a=1
                    if isacc[i]==1 :
                        count[labels[i]] += 1
                    num_samples[labels[i]]+=1
                    if isacc2[i]==1 :
                        count2[labels2[i]] += 1
                    num_samples2[labels2[i]]+=1
                    if self.mode=='validation' and False:
                        if labels[i] == 1 and output_numpy[i,-1]>0.99:
                            I = np.array(input[i, :, :, :].cpu().numpy() * 255, np.uint8).transpose(1, 2, 0) [:,:, [2, 1, 0]]
                            cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train1/abnor/abnor_' +
                                        names[i].split('/')[-1].split('.')[0]+'.jpg',I)
                        if labels[i] == 0 and output_numpy[i,0]>0.99:
                            I = np.array(input[i, :, :, :].cpu().numpy() * 255, np.uint8).transpose(1, 2, 0) [:,:, [2, 1, 0]]
                            cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train1/nor/nor_' +
                                        names[i].split('/')[-1].split('.')[0]+'.jpg',I)
                if i_batch%500==0 and i_batch>1:
                    #print(count[:self.cls_num].sum() / num_samples[:self.cls_num].sum(), np.mean(AA))
                    self.logger.info('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),
                                                                                    count.sum(),len(self.validationdataset),
                                                                                    1.0*count/num_samples))
                    self.logger.info('subcls,corret/tot:{}/{},current_acc:{}'.format(count2.sum(),len(self.validationdataset),
                                                                                    1.0*count2/num_samples2))
        # F=F[0:-1:100]
        # C=C[0:-1:100]
        # writer.add_embedding(torch.cat(F,0),
        #     metadata=torch.cat(C,0),
        #     #label_img=input,
        #     global_step=self.epoch)
        #self.logger.info(count[:self.cls_num].sum()/num_samples[:self.cls_num].sum(),np.mean(AA))
        #LL=np.array(LL)
        #np.save(self.savingnpy, LL)
        Matrix=np.array(Matrix)
        #Matrix2=np.array(Matrix2)
        self.logger.info(Matrix)
        #self.logger.info(Matrix2.round(2))
        #print(options['training']['save_prefix'])
        result=count2/num_samples2
        re_all=count2.sum()/num_samples2.sum()
        self.logger.info('-'*21)
        self.logger.info('All sub acc:'+str(re_all))
        self.logger.info('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(len(result)):
            self.logger.info('{:<10}|{:>10}'.format(settings2d.allsubtype[i], result[i]))
        self.logger.info('-'*21)
        return count/num_samples,count[:self.cls_num].sum()/num_samples[:self.cls_num].sum(),re_all