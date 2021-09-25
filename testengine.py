from torch.autograd import Variable
import torch
import time,tqdm,shutil
import torch.optim as optim
from datetime import datetime, timedelta
#from data import LipreadingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import Dmap_test_dataset,NCPJPGtestDataset,NCPJPGtestDataset_new,IndtestDataset,DBZ_test_dataset
import os, cv2,pdb
import toml
from models.net2d import densenet121,densenet161,resnet152,resnet152_plus,resnet152_R,resnet50
import numpy as np
#from models.g_cam import GuidedPropo
import matplotlib as plt
KEEP_ALL=False
SAVE_DEEP=True
import argparse
from for_dbz_pre.settings2d import *
from torch.cuda.amp import autocast

def _validate(modelOutput, labels, length,topn=1):
    modelOutput=list(np.exp(modelOutput.cpu().numpy())[:length,-1])#for covid19
    #pos_count=np.sum(np.array(modelOutput)>0.5)

    modelOutput.sort()
    averageEnergies = np.mean(modelOutput[-topn:])
    iscorrect = labels.cpu().numpy()==(averageEnergies>0.5)
    pred=(averageEnergies>0.5)
    return averageEnergies,iscorrect,pred

def _validate_cp(modelOutput, labels, length,topn=1):
    averageEnergies = np.exp(modelOutput.cpu().numpy()[:length, :]).mean(0)
    pred = np.argmax(averageEnergies)
    iscorrect = labels.cpu().numpy() == pred
    return averageEnergies.tolist(), iscorrect, pred

def _validate_ind(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    modelOutput=np.exp(modelOutput.cpu().numpy())
    cppro=np.max(modelOutput[:,1:3],-1)
    healthypre=modelOutput[:,0]
    ncp_pre = modelOutput[:, -1]
    modelOutput=np.stack([healthypre,cppro,ncp_pre],-1)
    for i in range(0,modelOutput.shape[1]):
        t = modelOutput[:length, i].tolist() # for covid19
        t.sort()
        if i==0:
            averageEnergies.append(np.mean(t[-topn*2:]))
        else:
            averageEnergies.append(np.mean(t[-topn:]))
    averageEnergies = averageEnergies / np.sum(averageEnergies, keepdims=True)
    pred=np.argmax(averageEnergies)
    label=labels.cpu().numpy()
    iscorrect = label == pred
    return averageEnergies.tolist(), [iscorrect],pred


def _validate_healthy_or_not(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    averageEnergies2=[]
    modelOutput=np.exp(modelOutput.cpu().numpy())
    illpro=np.mean(modelOutput[:,1:],1)
    healthypre=modelOutput[:,0]
    modelOutput=np.stack([healthypre,illpro],-1)
    for i in range(0,modelOutput.shape[1]):
        t = modelOutput[:length, i].tolist() # for covid19
        t.sort()
        if i==0:
            averageEnergies.append(np.mean(t[-1:]))
        else:
            averageEnergies2.append(np.mean(t[-topn:]))
    averageEnergies2=np.max(averageEnergies2)
    averageEnergies=np.array([averageEnergies[0],averageEnergies2])
    averageEnergies = averageEnergies / np.sum(averageEnergies, keepdims=True)
    pred=np.argmax(averageEnergies)
    if pred >=1:
        pred=1
    else:
        pred=0
    label=labels.cpu().numpy()
    if label>=1:
        label=1
    else:
        label=0
    iscorrect = label == pred
    return averageEnergies.tolist(), [iscorrect],pred

def _validate_cap_covid(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    output=np.exp(modelOutput.cpu().numpy())[:length, [1,3]]
    output=output/np.sum(output,1,keepdims=True)
    for i in range(output.shape[1]):
        t = output[:,i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        averageEnergies.append(np.mean(t[-topn:]))
    pred=np.argmax(averageEnergies)
    label=labels.cpu().numpy()
    if label==1:
        label=0
    else:
        label=1
    iscorrect = label == pred
    return averageEnergies, [iscorrect],pred

def _validate_hxnx_covid(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    output=np.exp(modelOutput.cpu().numpy())[:length, [2,3]]
    output = output / np.sum(output, 1, keepdims=True)
    for i in range(output.shape[1]):
        t = output[:,i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        averageEnergies.append(np.mean(t[-topn:]))
    pred=np.argmax(averageEnergies)
    averageEnergies=averageEnergies/np.sum(averageEnergies)
    label=labels.cpu().numpy()
    if label==2:
        label=0
    else:
        label=1
    iscorrect = label == pred
    return averageEnergies.tolist(), [iscorrect],pred

def _validate_multicls(modelOutput, labels,length,topn=3,healthy_cls=-1,topacc=1):
    averageEnergies=[]
    topn=[3,3,10,3,3]
    for i in range(0,modelOutput.shape[1]):
        if isinstance(modelOutput,np.ndarray):
            t =modelOutput[:length, i].tolist() # for covid19
        else:
            t =(modelOutput.cpu().numpy())[:length, i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        if True:
            if i==healthy_cls:
                averageEnergies.append(np.mean(t[-topn[i]:]))#
            else:
                averageEnergies.append(np.mean(t[-topn[i]:]))
        else:
            if i==healthy_cls:
                averageEnergies.append(np.mean(t[topn]))#
            else:
                averageEnergies.append(np.mean(t[topn]))
    #averageEnergies[0]= np.mean(averageEnergies[0:2])
    #averageEnergies[3]=np.sum(averageEnergies[1:])
    averageEnergies=averageEnergies/np.sum(averageEnergies)

    pred=np.argmax(averageEnergies)
    iscorrect = labels.cpu().numpy() == pred
    if topacc>1:
        pred=np.argsort(averageEnergies)[-topacc:].tolist()
        iscorrect=labels in pred
    return averageEnergies.tolist(), iscorrect,pred
def _validate_multicls2(modelOutput, labels,length,topn=3,healthy_cls=-1,topacc=1):
    averageEnergies=[]
    topn=[3]*12
    for i in range(0,modelOutput.shape[1]):
        if isinstance(modelOutput,np.ndarray):
            t =modelOutput[:length, i].tolist() # for covid19
        else:
            t =(modelOutput.cpu().numpy())[:length, i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        if True:
            if i==healthy_cls:
                averageEnergies.append(np.mean(t[-topn[i]:]))#
            else:
                averageEnergies.append(np.mean(t[-topn[i]:]))
        else:
            if i==healthy_cls:
                averageEnergies.append(np.mean(t[topn]))#
            else:
                averageEnergies.append(np.mean(t[topn]))
    #averageEnergies[0]= np.mean(averageEnergies[0:2])
    #averageEnergies[3]=np.sum(averageEnergies[1:])
    averageEnergies=averageEnergies/np.sum(averageEnergies)

    pred=np.argmax(averageEnergies)
    iscorrect = labels.cpu().numpy() == pred
    if topacc>1:
        pred=np.argsort(averageEnergies)[-topacc:].tolist()
        iscorrect=labels in pred
    return averageEnergies.tolist(), iscorrect,pred

def _validate_x_ornot(modelOutput, labels,length,x,topn=3):
    averageEnergies=[]
    target=modelOutput[:,x]
    elsecls=np.mean(modelOutput[:,x+1:],-1)
    modelOutput=np.stack([elsecls,target],-1)
    for i in range(0,modelOutput.shape[1]):
        if isinstance(modelOutput,np.ndarray):
            t =(modelOutput)[:length, i].tolist() # for covid19
        else:
            t =(modelOutput.cpu().numpy())[:length, i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        if topn>0:
            averageEnergies.append(np.mean(t[-topn:]))
        else:
            averageEnergies.append(np.mean(t[topn]))
    #averageEnergies[0]= np.mean(averageEnergies[0:2])
    #averageEnergies[3]=np.sum(averageEnergies[1:])
    averageEnergies=averageEnergies/np.sum(averageEnergies)

    pred=np.argmax(averageEnergies)
    labels=(labels==x)*1.0
    iscorrect = labels.cpu().numpy() == pred

    return averageEnergies.tolist(), iscorrect,pred

class Validator():
    def __init__(self, options, mode,model,savenpy=None,args=None,logger=None):
        self.logger=logger
        self.options=options
        self.USE_SUB=options['model']['subcls']
        self.R=options['general']['clinic']
        self.cls_num=options['general']['class_num']
        self.use_plus=options['general']['use_plus']
        self.use_3d = options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice = options['general']['use_slice']
        self.asinput = options['general']['plus_as_input']
        mod=options['general']['mod']
        #datalist = args.imgpath
        #masklist =args.maskpath
        self.savenpy = savenpy
        if mod=='healthy':
            f='data/lists/reader_healthy_vs_ill.list'
        elif mod=='cap':
            f = 'data/lists/reader_cap_vs_covid.list'
        elif mod=='AB-in':
            f = 'data/lists/reader_influenza_vs_covid.list'
            f = 'data/test_suyuan.list'
        elif mod=='ind':
            f = 'data/lists/ind_list_no_seg.list'
        elif mod=='xct':
            f = 'data/lists/testlist_xct.list'
        elif mod=='mosmed':
            f = 'data/test_MosMed.list'
        elif mod=='dbz':
            f='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/val2.json'
        elif mod=='reader1':
            f='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/reader1.json'
        elif mod=='reader2':
            f='/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/reader2.json'
        else: 
            f = 'data/testlist_ct_only.list'
            #f = 'data/test_suyuan.list'

        self.model=model
        self.mod=mod

        if mod=='ind':
            self.validationdataset = IndtestDataset(options[mode]["data_root"],
                                                    options[mode]["padding"],
                                                    f,cls_num=self.cls_num,mod=options['general']['mod'],
                                                    options=options)
        elif mod=='dbz' or mod=='reader2' or mod=='reader1':
            self.validationdataset=DBZ_test_dataset(options[mode]["data_root"],
                                                           options[mode]["padding"],
                                                            f,cls_num=self.cls_num,mod=options['general']['mod'],
                                                           options=options,logger=logger,one_else=options['one_else'])
            
            # self.validationdataset=Dmap_test_dataset(options[mode]["data_root"],
            #                                                options[mode]["padding"],
            #                                                 f,cls_num=self.cls_num,mod=options['general']['mod'],
            #                                                options=options,logger=logger,one_else=options['one_else'])
        else:
            self.validationdataset = NCPJPGtestDataset_new(options[mode]["data_root"],
                                                           options[mode]["padding"],
                                                            f,cls_num=self.cls_num,mod=options['general']['mod'],
                                                           options=options)

        self.topk=options['test']['topk']
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
            self.validationdataset,
            batch_size=1,
            shuffle=True,
            num_workers=2,
            drop_last=False
        )
        self.mode = mode
        self.epoch = 0
    #@autocast()
    def __call__(self):
        self.epoch += 1
        #f=open('suspective.list','w')
        with torch.no_grad():
            self.logger.info("Starting {}...".format('V test!'))
            count = np.zeros((self.cls_num + self.use_plus * 2*(1-self.asinput)))
            Matrix = np.zeros((self.cls_num, self.cls_num))
            count2 = np.zeros((len(allsubtype)))
            count25=np.zeros((len(allsubtype)))
            Matrix2 = np.zeros((len(allsubtype), len(allsubtype)))
            #Matrix25 = np.zeros((16, 16))
            if self.cls_num>2:
                if self.mod=='healthy':
                    validator_function=_validate_healthy_or_not#win0
                elif self.mod== 'cap':
                    validator_function = _validate_cap_covid
                elif self.mod== 'AB-in':
                    validator_function = _validate_hxnx_covid
                elif self.mod=='ind':
                    validator_function = _validate_multicls
                elif self.mod=='xct':
                    validator_function = _validate_cap_covid
                elif self.mod=='reader2':
                    validator_function = _validate_x_ornot
                else:
                    validator_function = _validate_multicls
            else:
                validator_function = _validate_cp
            self.model.eval()
            LL = []
            LL2=[]
            GG=[]
            AA=[]
            if (self.usecudnn):
                net = nn.DataParallel(self.model).cuda()
            num_samples = np.zeros((self.cls_num + self.use_plus * 2*(1-self.asinput)))
            num_samples2 = np.zeros((len(allsubtype) + self.use_plus * 2*(1-self.asinput)))
            tic=time.time()
            X=[]
            Y=[]
            Z=[]
            P=[]
            N=[]
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda().float()
                labels = Variable(sample_batched['label']).cuda()
                labels2 = Variable(sample_batched['label2']).cuda()
                if self.use_plus:
                    age = Variable(sample_batched['age']).cuda()
                    gender = Variable(sample_batched['gender']).cuda()
                    pos=Variable(sample_batched['pos']).cuda()
                if self.options['use_rc']:
                    r_feature = Variable(sample_batched['rfeature']).cuda()
                    c_feature = Variable(sample_batched['cfeature']).cuda()
                    r_feature=r_feature.permute(1,0,2)
                    c_feature=c_feature.permute(1,0,2)
                name =sample_batched['length'][0]
                valid_length=sample_batched['length'][1]
                
                rs=input.shape
                input=input.squeeze(0)
                input=input.permute(1,0,2,3)
                
                if input.shape[0]<3:
                    self.logger.info(name)
                    self.logger.info(str(input.shape[0]))
                    continue
                if self.options['model']['forest']:
                    #loss,loss1,loss2= net.module(input)
                    if self.options['use_rc']:
                        outputs,output2,deep_feature=net.module.get_output(input,r_feature,c_feature)
                    else:   
                        outputs,output2,deep_feature=net.module.get_output(input)
                else:
                    if not self.use_plus:
                        if self.USE_SUB:
                            if self.R:
                                outputs,deep_feature,output2 = net(input,features,False)##here
                            else:
                                outputs,deep_feature,output2 = net(input,False)##here
                        else:
                            outputs,deep_feature = net(input,False)
    
                    else:
                        if self.asinput:
                            outputs, _, _, _, deep_feaures = net(input,pos,gender,age)
                        else:
                            outputs, out_gender, out_age,out_pos,deep_feaures = net(input)
                if SAVE_DEEP and self.options['model']['forest']:
                    deep_feature=deep_feature.cpu().numpy()
                    I_r=input.cpu().numpy()[:]
                    X.append(deep_feature)
                    Z.append(name)
                    Y.append(labels2.cpu().numpy()[0][0])
                if KEEP_ALL:
                    all_numpy=outputs.cpu().numpy()[:valid_length,1]
                    np.save('multi_period_scores/npys_re/'+name[0].split('/')[-1]+'.npy',all_numpy)

                
                #pdb.set_trace()
                
                if not self.mod=='reader2':
                    (vector, isacc,pos_count) = validator_function(outputs, labels,valid_length,self.topk,)
                    #output2=output2[:,:5]
                    (output_numpy2, isacc2,pos_count2) = _validate_multicls2(output2, labels2,valid_length,self.topk,
                                                                    )
                    (output_numpy25, isacc25,pos_count25) = _validate_multicls2(output2, labels2,valid_length,
                                                               topacc=2)
                    
                else:
                    (vector, isacc,pos_count) = validator_function(outputs, labels,valid_length,0,self.topk)
                    (output_numpy2, isacc2,pos_count2) = _validate_multicls2(output2, labels2,valid_length,0,self.topk)
                    
                output_numpy = vector
                label_numpy = labels.cpu().numpy()[0, 0]
                label_numpy2 = labels2.cpu().numpy()[0, 0]
                if self.mod=='healthy':
                    if label_numpy>=1:
                        label_numpy=1
                    else:
                        label_numpy=0
                elif self.mod=='cap' or self.mod=='xct':
                    if label_numpy==1:
                        label_numpy=0
                    else:
                        label_numpy=1
                elif self.mod=='AB-in':
                    if label_numpy==2:
                        label_numpy=0
                    else:
                        label_numpy=1
                elif self.mod=='reader2':
                    if label_numpy==0:
                        label_numpy=1
                    else:
                        label_numpy=0
                #print(i_batch)
                # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    LL.append([name[0]]+ output_numpy+[label_numpy])
                    LL2.append([name[0]]+ output_numpy2+[label_numpy2])
                    Matrix[label_numpy, pos_count] += 1
                    Matrix2[label_numpy2, pos_count2] += 1
                    #Matrix25[]+=1
                    #if isacc[i]==0:
                    #    print(name[0]+'\t'+str(label_numpy)+'\t'+str(pos_count)+'\t'+str(vector))
                    if isacc[i] == 1:
                        count[label_numpy] += 1
                    if isacc2[i] == 1:
                        count2[label_numpy2] += 1
                    if not self.mod=='reader2':
                        if isacc25 == 1:
                            count25[label_numpy2] += 1
                    num_samples[label_numpy] += 1
                    num_samples2[label_numpy2] += 1
                    if i_batch%1000==0 and i_batch>1:
                        #print(count[:self.cls_num].sum() / num_samples[:self.cls_num].sum(), np.mean(AA))
                        self.logger.info('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),
                                                                                      count.sum(),len(self.validationdataset),
                                                                                       1.0*count/num_samples))
                        self.logger.info('subcls,corret/tot:{}/{},current_acc:{}'.format(count2.sum(),len(self.validationdataset),
                                                                                       1.0*count2/num_samples2))
                if False and self.mod=='all':
                    if labels[0] == 3:
                        prob = torch.exp(outputs)[:,-1].detach().cpu().numpy()
                        #prob_idx=np.argsort(prob)
                        for idd in range(outputs.shape[0]):
                            I = np.array(input[idd, :, :, :].cpu().numpy() * 255, np.uint8).transpose(1, 2, 0) [:,:, [2, 1, 0]]
                            J = I[I[:, :, 2] == 255, 1].mean()
                            if  J>50.5:
                                cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train2/abnor/abnor_' +
                                            name[0].split('/')[-1].split('.')[0]+'_'+str(idd)+'.jpg',I)
                            if J<27.5:
                                cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train2/nor/nor_' +
                                            name[0].split('/')[-1].split('.')[0]+'_'+str(idd)+'.jpg',I)


        self.logger.info(maintype)
        #print(count[:self.cls_num].sum() / num_samples[:self.cls_num].sum(),np.mean(AA))
        LL = np.array(LL)
        LL2=np.array(LL2)
        self.logger.info(Matrix)
        #self.logger.info(allsubtype)
        #self.logger.info(Matrix2)
        np.save(self.savenpy, LL)
        np.save(self.savenpy+'level2.npy', LL2)
        print(f'saving to {self.savenpy}')
        if SAVE_DEEP and False:
            X=np.array(X)
            Y=np.array(Y)
            Z = np.array(Z)
            np.save(os.path.join('saves','X_dbz_temp_severe.npy'),X)
            np.save(os.path.join('saves', 'Y_dbz_temp_severe.npy'), Y)
            np.save(os.path.join('saves', 'Z_dbz_temp_severe.npy'), Z)
        if self.use_plus and not self.asinput:
            GG = np.array(GG)
            AA=np.array(AA)
            np.save('gender.npy', GG)
            np.save('age.npy', AA)
        toc=time.time()
        self.logger.info((toc-tic)/self.validationdataloader.dataset.__len__())

        result=count2/num_samples2
        re_all=count2.sum()/num_samples2.sum()
        self.logger.info('-'*21)

        self.logger.info('All sub acc:'+str(re_all))
        self.logger.info('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(len(result)):
            self.logger.info('{:<10}|{:>10}'.format(allsubtype[i], result[i]))
        self.logger.info('-'*21)
        if not self.mod=='reader2':
            result5=count25/num_samples2
            re_all5=count25.sum()/num_samples2.sum()
            self.logger.info('-'*21)
            self.logger.info('All sub acc:'+str(re_all5))
            self.logger.info('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
            for i in range(len(result5)):
                self.logger.info('{:<10}|{:>10}'.format(allsubtype[i], result5[i]))
            self.logger.info('-'*21)

        return count / num_samples, count[:self.cls_num].sum() / num_samples[:self.cls_num].sum()

    def age_function(self, pre, label):
        pre=pre.cpu().numpy().mean()* 90
        label=label.cpu().numpy()
        return np.mean(pre-label),pre


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--deepsave", help="A path to save deepfeature", type=str,
                        # default='re/cap_vs_covid.npy')
                        default='deep_f')
    parser.add_argument("-e", "--exclude_list",
                        help="A path to a txt file for excluded data list. If no file need to be excluded, "
                             "it should be 'none'.", type=str,
                        default='none')
    parser.add_argument("-v", "--invert_exclude", help="Whether to invert exclude to include", type=bool,
                        default=False)
    parser.add_argument("-k", "--topk", help="gpuid", type=int,
                        default=5)
    parser.add_argument("-s", "--savenpy", help="gpuid", type=str,
                        default='temp.npy')
    args = parser.parse_args()
    os.makedirs(args.deepsave, exist_ok=True)

    print("Loading options...")
    with open('test.toml', 'r') as optionsFile:
        options = toml.loads(optionsFile.read())

    if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']

    torch.manual_seed(options["general"]['random_seed'])

    # Create the model.
    if options['general']['use_plus']:
        model = resnet152_plus(options['general']['class_num'])
    else:
        model = resnet152(options['general']['class_num'])
    if 'R' in options['general'].keys():
        model = resnet152_R(options['general']['class_num'])
    pretrained_dict = torch.load(options['general']['pretrainedmodelpath'])
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    print('matched keys:', len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    tester = Validator(options, 'test',model,options['validation']['saves'],args)

    result, re_all = tester()
    print (tester.savenpy)
    print('-' * 21)
    print('All acc:' + str(re_all))
    print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
    for i in range(result.shape[0]):
        print('{:<10}|{:>10}'.format(i, result[i]))
    print('-' * 21)



if __name__ == "__main__":
    main()
