import json,os
datas=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/train.json','r'))
for one in list(datas.keys()):
    thisone=datas[one]
    segpath=thisone['newpath'].replace('newdisk3/data_for2d','newdisk2/reg/transform').replace('.nii','.seg.nii')
    if not os.path.exists(segpath):
        datas.pop(one)

json.dump(datas,open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/train_f.json','w'))