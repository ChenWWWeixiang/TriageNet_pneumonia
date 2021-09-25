import os,random,glob
import argparse,sys
sys.path.append('/mnt/data9/Lipreading-DenseNet3D-master')
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-p", "--path", help="A list of paths to jpgs for seperate",
                    type=str,
                    default=['/mnt/data9/covid_detector_jpgs/pos2_train',
                            #'/mnt/data9/covid_detector_jpgs/no_seg/healthy',
                           # '/mnt/data9/covid_detector_jpgs/no_seg/healthy2',
                           # '/mnt/data9/covid_detector_jpgs/no_seg/covid2',
                             ])
parser.add_argument("-t", "--train_txt",
                    help="train list output path",
                    type=str,
                    default='data/txt/pos_map_train.txt')
args = parser.parse_args()
if isinstance(args.path,str):
    path=eval(args.path)
else:
    path=args.path
f1 = open(args.train_txt, 'w')

for ipath in path:
    cnt = 0
    for type1 in  os.listdir(ipath):
        for type2 in os.listdir(os.path.join(ipath,type1)):
            for files in glob.glob(os.path.join(ipath,type1,type2,'*dmap.jpg')):
                #set_name=ipath.split('/')[-1]
                #name=os.path.join(ipath,type1,type2,files)
                f1.writelines(files.replace('dmap','') + '\n')
