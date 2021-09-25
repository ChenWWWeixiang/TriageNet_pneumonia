import os,glob,random
#data_path='/home/cwx/extra/covid_project_data'
#seg_path='/home/cwx/extra/covid_project_segs'
seg_path = '/mnt/data9/suyuan_nii/lung'
data_path = '/mnt/data9/suyuan_nii'

f2=open('none.list','w')
f=open('test_suyuan.list','w')
def set_it(all_files,set_name):
    #person_name=[item.split('/')[-1].split('_')[1] for item in all_files]
    #person_name = [item.split('/')[-1].split('_')[0] for item in all_files]
    #person_name=list(set(person_name))
    #l = len(person_name)
    for i,name in enumerate(all_files):
        all_ct=name
        f.writelines(all_ct.split('lung/lung_')[0]+all_ct.split('lung/lung_')[-1]+','+all_ct +'\n')




all_files=glob.glob(os.path.join(seg_path,'*.nii'))
random.shuffle(all_files)
set_it(all_files,'1')


