from __future__ import print_function
from models.Dense3D import Dense3D
import torch,argparse
import toml,logging
from training import Trainer
from testengine import Validator
from validation import Validator as Validator2
import torch.nn as nn
import os
from models.net2d import densenet161,resnet152,resnet152_plus,resnet152_R,ResLSTM,ForestCLF,ForestCLF_RC
import warnings
#basically, the best params: n-1600-5
warnings.filterwarnings("ignore")
from for_dbz_pre.settings2d import *
import argparse


fx=open('resultslist.csv','a+')
#torch.cuda.amp.autocast=True
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

log_path="logs/log-{}-rc:{}-forest:{}.txt".format(PRE_FORSAVE,options['use_rc'],options['model']['forest'])
if os.path.exists(log_path):
    os.remove(log_path)
handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
 
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)
if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    #print("Running cudnn benchmark...")
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
#options["general"]['gpuid']=GPU_SETTING
os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']
    
torch.manual_seed(options["general"]['random_seed'])

#torch.backends.cudnn.benchmark = True
#Create the model.
if options['general']['use_3d']:
    model = Dense3D(options)
elif options['general']['use_slice']:
    if options['general']['use_plus']:
        model = resnet152_plus(options['general']['class_num'],asinput=options['general']['plus_as_input'],
                               USE_25D=False)
    elif  options['general']['use25d']:
        model = ResLSTM(options['general']['class_num'])
    else:
        model = resnet152(options['general']['class_num'],setting=options)#vgg19_bn(2)#squeezenet1_1(2)
    if  options['general']['clinic']:
        model=resnet152_R(options['general']['class_num'],setting=options)
    if  options['model']['forest']:
        if options['use_rc']:
            if options['use_v2']:
                from models.net2d import ForestCLF_RC2 as ForestCLF_RC
                model=ForestCLF_RC(options['model']['nt'],model,num_features= options['model']['nf'],
                                num_of_cls= options['model']['nc'],num_of_cls2=options['model']['nc2'])
            else:
                model=ForestCLF_RC(options['model']['nt'],model,num_features= options['model']['nf'],num_of_cls= options['model']['nc'],
                    num_of_cls2=options['model']['nc2'])
        else:
            from models.net2d import ForestCLF_2 as ForestCLF_RC
            #from models.net2d import ForestCLF_dmap as ForestCLF_RC
            model=ForestCLF_RC(options['model']['nt'],model,num_features= options['model']['nf'],
                            num_of_cls= options['model']['nc'],num_of_cls2=options['model']['nc2'])
    
else:
    model=densenet161(2)
#logger=logging.getLogger(__name__)


#model = torch.nn.parallel.DataParallel(model).cuda()
sepoch=options["training"]["startepoch"]
if(options["general"]["loadpretrainedmodel"]):
    # remove paralle module
    if os.path.exists(options["general"]["pretrainedmodelpath"].format(PRE_FORSAVE)):
        pretrained_dict = torch.load(options["general"]["pretrainedmodelpath"].format(PRE_FORSAVE))
        # load only exists weights
        model_dict = model.state_dict()
        if options['model']['forest']: 
            model.cc=pretrained_dict['cc']
            model.idx=pretrained_dict['idx']
        #sepoch=pretrained_dict['epoch']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        print('matched keys:',len(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        if options['model']['forest']: 
            model.init_params()
            #model.fixed_groups_params()
else:
    if options['model']['forest']: 
        model.init_params()
       # model.fixed_groups_params()

if(options["general"]["usecudnn"]):        
    torch.cuda.manual_seed(options["general"]['random_seed'])
    torch.cuda.manual_seed_all(options["general"]['random_seed'])

if True:
    trainer = Trainer(options,model,logger,sum_path=log_path)
if(options["validation"]["validate"]):
    if options['general']['mod']=='slice':
        validator = Validator2(options, 'validation',model,savenpy=options["validation"]["saves"].format(PRE_FORSAVE))
    else:
        #validator = Validator(options, 'validation',model,savenpy=options["validation"]["saves"].format(PRE_FORSAVE),logger=logger)
        validator = Validator2(options, 'validation',model,savenpy=options["validation"]["saves"].format(PRE_FORSAVE),logger=logger)
if(options['test']['test']):   
    tester = Validator(options, 'validation',model,savenpy=options["validation"]["saves"].format(PRE_FORSAVE),logger=logger)
flag=0
for epoch in range(sepoch, options["training"]["epochs"]):
    if(options["training"]["train"]):
        if flag:
            trainer(epoch,refine_weight,flag)
            logger.info('get a new w'+str(refine_weight))
        else:
            a=1
            trainer(epoch)
    if (options["validation"]["validate"]) and (epoch%5==0):
        result,re_all,re_all_sub = validator(trainer.writer)
        #trainer.ScheduleLR(result.min())
        #logger.info(options['training']['save_prefix'].format(PRE_FORSAVE))
        print('validation')
        logger.info('-'*21)
        logger.info('All acc:'+str(re_all))
        logger.info('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(len(result)):
            logger.info('{:<10}|{:>10}'.format(maintype[i], result[i]))
        logger.info('-'*21)
        refine_weight=1-result
        refine_weight=refine_weight/refine_weight.sum()
        trainer.writer.add_scalar('val acc all',re_all,trainer.tot_iter)
        #flag=1

    if(options['test']['test']) and epoch%20==0:
        print('here!')
        result,re_all = tester()
        logger.info('-'*21)
        logger.info('All acc:' + str(re_all))
        logger.info('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(len(result)):
            logger.info('{:<10}|{:>10}'.format(maintype[i], result[i]))
        logger.info('-'*21)

        print('-'*21)
        print('All acc:' + str(re_all))
        print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(len(result)):
            print('{:<10}|{:>10}'.format(maintype[i], result[i]))
        print('-'*21)
try:
    x=[]
    for i in range(5):
        x.append((i in args.cls_list)*1.0)
    fx.writelines('{},{},{},{},{},{},{},{},{},{}\n'.format(x[0],x[1],x[2],x[3],x[4],args.m,args.n,args.k,re_all,re_all_sub))
    trainer.writer.close()
    fx.close()
except:
    a=1
logger.info(options['training']['save_prefix'].format(PRE_FORSAVE))