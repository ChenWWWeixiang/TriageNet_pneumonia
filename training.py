from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data.dataset import DBZ_V_dataset, NCPDataset,NCP2DDataset,NCPJPGDataset,NCPJPGDataset_new,NCPJPGtestDataset_new,DBZ_dataset
from torch.utils.data import DataLoader
#from data.dataset import Dmap_dataset
from tensorboardX import SummaryWriter
import torch.nn as nn
import os,cv2,shutil
import pdb
import math
from torch.cuda.amp import autocast
import torch.cuda.amp as amp
from for_dbz_pre.settings2d import *

class NLLSequenceLoss(torch.nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = torch.nn.NLLLoss(reduction='none')

    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i,], target).unsqueeze(1))
        loss = torch.cat(loss, 1)
        # print('loss:',loss)
        mask = torch.zeros(loss.size(0), loss.size(1)).float().cuda()

        for i in range(loss.size(0)):
            L = min(mask.size(1), length[i])
            mask[i, :L - 1] = 1.0
        # print('mask:',mask)
        # print('mask * loss',mask*loss)
        loss = (loss * mask).sum() / mask.sum()
        return loss

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{:0>2} hrs, {:0>2} mins, {:0>2} secs".format(hours, minutes, seconds)

def output_iteration(loss, i, time, totalitems,logger):

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    
    logger.info("Iteration: {:0>8},Elapsed Time: {},Estimated Time Remaining: {},Loss:{}".format(i, timedelta_string(time), timedelta_string(estTime),loss))

class Trainer():
    def __init__(self, options,model,logger,sum_path):
        sum_path="runs/tb-{}-rc:{}-forest:{}".format(PRE_FORSAVE,options['use_rc'],options['model']['forest'])
        self.options=options
        self.tot_iter = 0
        self.writer = SummaryWriter(log_dir=sum_path) 
        self.logger=logger
        self.USE_SUB=options['model']['subcls']
        self.R=options['general']['clinic']
        self.cls_num=options['general']['class_num']
        self.use_plus=options['general']['use_plus']
        self.use_slice = options['general']['use_slice']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_3d=options['general']['use_3d']
        self.batchsize = options["input"]["batchsize"]
        self.use_lstm=options["general"]["use25d"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.learningrate = options["training"]["learningrate"]
        self.modelType = options["training"]["learningrate"]
        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        self.save_prefix = options["training"]["save_prefix"]
        self.asinput=options['general']['plus_as_input']
        self.USE_25D=options['general']['use25d']
        if options['general']['use_slice']:
            if self.USE_25D:
                self.trainingdataset = DBZ_V_dataset(options["training"]["index_root"],options["training"]["padding"],
                                                cls_num=self.cls_num,options=options,logger=self.logger,is_train=True)
            else:
                if options['general']['mod']=='dbz' or options['general']['mod']=='reader2':
                    self.trainingdataset=DBZ_dataset(options["training"]["data_root"],
                                                options["training"]["index_root"],
                                                options["training"]["padding"],
                                                True,cls_num=self.cls_num,mod=options['general']['mod'],options=options,logger=self.logger,
                                                one_else=options['one_else'])
                    # self.trainingdataset=Dmap_dataset(options["training"]["data_root"],
                    #                             options["training"]["index_root"],
                    #                             options["training"]["padding"],
                    #                             True,cls_num=self.cls_num,mod=options['general']['mod'],options=options,logger=self.logger,
                    #                             one_else=options['one_else'])
                else:
                    self.trainingdataset = NCPJPGDataset_new(options["training"]["data_root"],
                                                options["training"]["index_root"],
                                                options["training"]["padding"],
                                                True,cls_num=self.cls_num,mod=options['general']['mod'],options=options)
        else:
            if options['general']['use_3d']:
                self.trainingdataset = NCPDataset(options["training"]["index_root"],
                                                    options["training"]["padding"],
                                                    True,
                                                    z_length=options["model"]["z_length"])
            else:
                self.trainingdataset = NCP2DDataset(options["training"]["data_root"],
                                                    options["training"]["index_root"],
                                                    options["training"]["padding"],
                                                    True)
        weights = self.trainingdataset.make_weights_for_balanced_classes()
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(self.trainingdataset))

        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=options["input"]["batchsize"],
                                    #shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False,
                                    sampler=sampler,
                                    pin_memory=True,
                                    )
        self.scaler = amp.GradScaler()
        self.optimizer = optim.Adam(model.parameters(),lr = self.learningrate,amsgrad=True)
        #self.schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'max',
        #                                                         patience=3, factor=.3, threshold=1e-3, verbose=True)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.model=model
        if self.use_3d:
            self.criterion=self.model.loss()
        else:
            #criterion=nn.
            #w=torch.Tensor(self.trainingdataset.get_w()).cuda()
            #print(w)
            #w = torch.Tensor(w).cuda()
            self.criterion =nn.NLLLoss().cuda()#0.3,0.7
            self.criterion_m =nn.NLLLoss().cuda()
            if self.use_plus:
                self.criterion_age = nn.NLLLoss(ignore_index=-1).cuda()
                self.criterion_gender = nn.NLLLoss(ignore_index=-1).cuda()
                self.criterion_pos=nn.SmoothL1Loss().cuda()
        if self.use_lstm:
            self.criterion=NLLSequenceLoss()
        if(self.usecudnn):
            self.net = nn.DataParallel(self.model).cuda()
            self.criterion = self.criterion.cuda()

    def ScheduleLR(self,acc):
        self.scheduler.step(acc)
    def __call__(self,epoch,refine_weight=None,flag=False):
        #set up the loss function.
        self.net.train()
        startTime = datetime.now()
        self.logger.info("Starting training...")
        if flag==True:
            self.criterion_m=nn.NLLLoss(weight=torch.Tensor([refine_weight]).cuda().half())
        for i_batch, sample_batched in enumerate(self.trainingdataloader):
            self.optimizer.zero_grad()
            
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])
            labels2 = Variable(sample_batched['label2'])
            #features = Variable(sample_batched['features'])
        # name=sample_batched['name']
            if self.options['use_rc']:
                r_feature = Variable(sample_batched['rfeature']).cuda()
                c_feature = Variable(sample_batched['cfeature']).cuda()
                #pos=Variable(sample_batched['pos']).cuda()
            input = input.cuda()
            labels = labels.cuda()
            labels2 = labels2.cuda()
            if self.options['model']['forest']:
                if self.options['use_rc']:
                    loss,loss1,loss2,loss3,deep_feature= self.net(input,r_feature,c_feature,labels,labels2)
                else:
                    loss,loss1,loss2,loss3,deep_feature= self.net(input,labels,labels2)
                #self.net.module.get_pre1(labels)
                self.writer.add_scalar('Train Loss main', loss1, self.tot_iter)
                self.writer.add_scalar('Train Loss sub', loss2, self.tot_iter)
                self.writer.add_scalar('Train Loss weight', loss3, self.tot_iter)
                self.writer.add_scalar('Train Loss all', loss, self.tot_iter)
            else:
                if not self.use_plus:
                    if self.USE_SUB:
                        if self.R:
                            outputs,deep_feature,output2 = self.net(input,features,False)##here
                        else:
                            outputs,deep_feature,output2 = self.net(input,False)##here
                    else:
                        outputs,deep_feature = self.net(input,False)
                else:
                    if self.asinput:
                        outputs, _, _, _, deep_feaures = self.net(input,pos,gender,age)
                    else:
                        outputs,out_gender,out_age,out_pos,deep_feaures=self.net(input)
                if self.use_3d or self.use_lstm:
                    length = sample_batched['length']
                    loss1 = self.criterion(outputs.log_softmax(-1), length[0],labels.squeeze(1))
                    self.writer.add_scalar('Train Loss main', loss1, self.tot_iter)
                    loss2=self.criterion(output2.log_softmax(-1), length[0],labels2.squeeze(1))
                    self.writer.add_scalar('Train Loss sub', loss2, self.tot_iter)
                    loss=loss1*0.7+loss2*0.3
                    self.writer.add_scalar('Train Loss all', loss, self.tot_iter)
                    newpred = torch.zeros_like(outputs)
                    #for j, atype in enumerate(TYPEMAT):
                    #    newpred[:, j] = output2[:, atype].sum()
                    #loss+=0.2*nn.functional.mse_loss(newpred.log_softmax(-1), outputs)
                else:
                    if self.USE_SUB:
                        if True:
                            loss1 = self.criterion_m(outputs.log_softmax(-1), labels.squeeze(1))
                            loss2=  self.criterion(output2.log_softmax(-1), labels2.squeeze(1))
                            if torch.isnan(loss1):
                                print(self.tot_iter)
                           # losse=nn.functional.mse_loss(newpred.log_softmax(-1), outputs.log_softmax(-1))
                            #loss3=self.criterion(output2[:,:2].log_softmax(-1),outputs)
                            loss=loss1*0.7+loss2*0.3#+0.0*losse
                            self.writer.add_scalar('Train Loss main', loss1, self.tot_iter)
                            self.writer.add_scalar('Train Loss sub', loss2, self.tot_iter)
                            #self.writer.add_scalar('Train Loss con', losse, self.tot_iter)
                            self.writer.add_scalar('Train Loss all', loss, self.tot_iter)

                        else:
                            loss = self.criterion(outputs.log_softmax(-1), labels.squeeze(1))+ self.criterion(output2.log_softmax(-1), labels2.squeeze(1))
                    else:
                        loss = self.criterion(outputs.log_softmax(-1), labels.squeeze(1))
            self.scaler.scale(loss).backward()
            # 将梯度值缩放回原尺度后，优化器进行一步优化
            self.scaler.step(self.optimizer)

            # 更新scalar的缩放信息
            self.scaler.update()
            # loss.backward()
            # self.optimizer.step()
            self.scheduler.step()
            sampleNumber = i_batch * self.batchsize

            if(self.tot_iter % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(loss.cpu().detach().numpy(), sampleNumber, currentTime - startTime, len(self.trainingdataset),self.logger)
               
            self.tot_iter += 1


        self.logger.info("Epoch "+str(epoch)+"completed, saving state...")
        self.logger.info(self.use_3d)
        tt=self.model.state_dict()
        if self.options['model']['forest']:
            tt['cc']=self.model.cc
            tt['idx']=self.model.idx
        tt['epoch']=epoch
        torch.save(tt, "{}-rc:{}-forest:{}.pt".format(self.save_prefix.format(PRE_FORSAVE),
                self.options['use_rc'],self.options['model']['forest']))

