import os
root='/mnt/data9/covid_detector_jpgs/match_ind'
allfile=os.listdir(root)
for item in allfile:
    os.rename(os.path.join(root,item),os.path.join(root,item[:-1]))