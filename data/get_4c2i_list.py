import os,random,glob
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-p", "--path", help="A list of paths to jpgs for seperate",
                    type=str,
                    default= '/mnt/data9/covid_detector_jpgs/match_ind_sep')
parser.add_argument("-v", "--val_txt",
                    help="validation list output path",
                    type=str,
                    default='lists/ind_train.list')

args = parser.parse_args()
#path=['/mnt/data7/slice_test_seg/jpgs2']
f2 = open(args.val_txt, 'w')
path=args.path
All = []
for type in os.listdir(path):

    for person in os.listdir(os.path.join(path,type)):
        for scan in os.listdir(os.path.join(path,type,person)):
            All.append(os.path.join(path,type,person,scan))

for p in All:
    f2.writelines(p+'\n')

