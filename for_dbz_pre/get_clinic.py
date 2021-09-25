import os,xlrd,json
import pydicom,glob
import pandas as pd
import numpy as np 
from xlrd import xldate_as_tuple
from datetime import datetime
datapath='/mnt/newdisk3/raw_data/all'
alldir_raw=glob.glob(datapath+'/*/*/')
alldir=[item.split('/')[-2] for item in alldir_raw]

booknames=[
        "/mnt/newdisk3/raw_data/肺炎收集资料-病毒性.xlsx",
        "/mnt/newdisk3/raw_data/肺炎收集资料-细菌性.xlsx",
        "/mnt/newdisk3/raw_data/肺炎收集资料-真菌性肺炎.xlsx",
        "/mnt/newdisk3/raw_data/肺炎收集资料-衣原体+支原体.xlsx"]
train_list='for_dbz_pre/jsons/train.json'
eval_list='for_dbz_pre/jsons/val.json'
train_list=json.load(open(train_list,'r'))
eval_list=json.load(open(eval_list,'r'))
alls_list_temp = {**train_list, **eval_list}

newlist=dict()
def find_pid(diction, key):
    for a in diction.keys():
        if diction[a]['pid']==key:
            return a
    return -1
def find_innerid(diction,key):
    for a in diction.keys():
       # if diction[a]['clsII']=='covid19' or diction[a]['clsII']=='healthy':
        #    continue
        try:
            if key in diction[a]['path']:
                return a
        except:
            print(diction[a])
        if key.replace('_','/') in diction[a]['path']:
            return a
    return -1

cnt=0
for book in booknames:
    excels = xlrd.open_workbook(book)
    sheetNames = excels.sheet_names()
    for asheet in sheetNames:
        ##read part
        table = excels.sheet_by_name(asheet)
        data=np.array(table._cell_values)
        inner_id=data[2:,1]
        pid=data[2:,2]
        name=data[2:,3]
        gender=data[2:,4]
        data[data=='']=np.nan
        data[data==' ']=np.nan
        age=data[2:,5]
        basic_ill=data[2:,8:18].astype(np.float)
        first_time=[]
        hospital_time=[]
        out_time=[]
        for i in range(2,data.shape[0]):
            if not table.cell(i,18).value=='':
                y= datetime(*xldate_as_tuple(table.cell_value(i,18), 0))
                first_time.append(y.strftime('%Y-%m-%d'))
            else:
                first_time.append('')
            if not table.cell(i,19).value=='':
                y= datetime(*xldate_as_tuple(table.cell_value(i,19), 0))
                hospital_time.append(y.strftime('%Y-%m-%d'))
            else:
                hospital_time.append('')
            if not table.cell(i,20).value=='':
                y= datetime(*xldate_as_tuple(table.cell_value(i,20), 0))
                out_time.append(y.strftime('%Y-%m-%d'))
            else:
                out_time.append('')
        first_time=np.array(first_time)
        hospital_time=np.array(hospital_time)
        out_time=np.array(out_time)

        basic_sick=data[2:,23:28].astype(np.float)
        blood_normal=data[2:,28:44].astype(np.float)
        blood_2=data[2:,50:60].astype(np.float)
        print(blood_normal.shape[1])
        print(blood_2.shape[1])
        a=1
        ##write part
        for i in range(pid.shape[0]):
            keyx=find_pid(alls_list_temp,pid[i].upper())
            if keyx==-1:
                keys=find_pid(alls_list_temp,pid[i])
                if keyx==-1:
                    keys=find_pid(alls_list_temp,pid[i].lower())
                    #if asheet=='新型冠状病毒1009个（320个）':
                    #    continue#debug
                    if keyx==-1 :
                        keyx=find_innerid(alls_list_temp,inner_id[i])
                        if keyx==-1:
                            cnt+=1
                            print(asheet,pid[i],inner_id[i],'not found!')
                            continue
            newlist[keyx]=alls_list_temp[keyx]
            newlist[keyx]['basic_ill']=basic_ill[i,:].tolist()
            newlist[keyx]['basic_sick']=basic_sick[i,:].tolist()
            newlist[keyx]['blood_normal']=blood_normal[i,:].tolist()
            newlist[keyx]['blood_2']=blood_2[i,:].tolist()
            newlist[keyx]['name']=keyx
            newlist[keyx]['timetable']=[first_time[i],hospital_time[i],out_time[i]]
            newlist[keyx]['inner_id']=inner_id[i].tolist()
print('loss',cnt)

##re perform train val division
new_train=dict()
new_test=dict()

for item in newlist.keys():
    if item in train_list.keys():
        new_train[newlist[item]['pid']] = newlist[item]
        #new_train[newlist[item]]['name']=item
    else:
        new_test[newlist[item]['pid']] = newlist[item]
        #new_test[newlist[item]]['name']=item
newlist = {**new_train, **new_test}
json.dump(newlist,open('for_dbz_pre/jsons/c_all.json','w'))
json.dump(new_train,open('for_dbz_pre/jsons/c_train.json','w'))
json.dump(new_test,open('for_dbz_pre/jsons/c_val.json','w'))