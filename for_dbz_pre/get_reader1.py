import json,random,shutil,os
import SimpleITK as sitk
maintype=['virus','fungus','bacteria','chlamydia','mycoplasma',]
allsubtype= ['CMV', 'Coxsackie virus', 'H7N9', 'Respiratory syncytial','covid19']+\
    ['aspergillus', 'candida', 'cryptococcus', 'PCP']+\
 ['Acinetobacter bowman', 'Klebsiella', 'Pseudomonas aeruginosa', 'S. aureus', 'Streptococcus']+\
            ['chlamydia','mycoplasma',]
data=json.load(open('/mnt/data9/Lipreading-DenseNet3D-master/for_dbz_pre/jsons/val.json','r'))
nums = dict()
dest='/mnt/newdisk1/reader1'
os.makedirs(dest,exist_ok=True)
for item in data:
    try:
        clsII=allsubtype.index(data[item]['clsII'])
    except:
        continue
    if str(clsII) in nums.keys():
        nums[str(clsII)].append(item)
    else:
        nums[str(clsII)]=[item]
readerset=dict()
for i in range(3):
    for item in nums:
        idx=random.choice(nums[item])
        nums[item].pop(nums[item].index(idx))
        readerset[idx]=data[idx]
idx=random.choice(nums['4'])
nums['4'].pop(nums['4'].index(idx))
readerset[idx]=data[idx]
idx=random.choice(nums['4'])
nums['4'].pop(nums['4'].index(idx))
readerset[idx]=data[idx]

shuffle_list=list(readerset.keys())
random.shuffle(shuffle_list)
with open('answer1.txt','w') as f:
    for id, item in enumerate(shuffle_list):
        f.writelines(str(id)+','+readerset[item]['newpath']+','+str(allsubtype.index(readerset[item]['clsII']))+'\n')
        data=sitk.ReadImage(readerset[item]['newpath'])
        img = sitk.GetArrayFromImage(data)
        #img =np.transpose(img,(1,0,2,3))
        os.makedirs(os.path.join(dest,str(id)),exist_ok=True)
        for j in range(img.shape[0]):
            data=sitk.GetImageFromArray(img[j,:,:].astype('int16'))
            sitk.WriteImage(data,os.path.join(dest,str(id),'{}'.format(str(j))+'.dcm'))
        #shutil.copy(readerset[item]['newpath'],os.path.join(dest,str(id)+'.'+readerset[item]['newpath'].split('.')[-1]))