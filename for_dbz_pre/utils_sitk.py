import SimpleITK as sitk
import numpy as np
import cv2
import h5py,os

def get_resampled(input,outputsize,resampled_spacing=[0.6,0.6,1]):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkBSpline)

    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())
    #resampler.SetSize(must_shape.GetSize())
    resampler.SetSize(outputsize.tolist())
    moving_resampled = resampler.Execute(input)
    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)
    return  moving_resampled,moving_resampled_ar


def get_resampled_with_box(input,box,resampled_spacing=[1,1,1],l=True):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())
    resampler.SetSize((1000,1000,1000))
    input_ar=sitk.GetArrayFromImage(input)
    input_box_map=np.zeros_like(input_ar)
    input_box_map[box[2]:box[5],box[1]:box[4],box[0]:box[3]]=255
    mask_map=sitk.GetImageFromArray(input_box_map)
    mask_map.CopyInformation(input)

    moving_resampled = resampler.Execute(input)
    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)
    try:
        xx, yy, zz = np.where(moving_resampled_ar > 0)
        resampled_data_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    except:
        xx, yy, zz,ee = np.where(moving_resampled_ar > 0)
        resampled_data_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]

    resampled_data = sitk.GetImageFromArray(resampled_data_ar)

    mask_map_re = resampler.Execute(mask_map)
    mask_map_ar = sitk.GetArrayFromImage(mask_map_re)
    if len(mask_map_ar.shape)==4:
        mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(), 0]
    else:
        mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    xx,yy,zz=np.where(mask_map_ar>=128)
    box_new=[xx.min(),yy.min(),zz.min(),xx.max(),yy.max(),zz.max()]#z,y,x
    return  resampled_data,resampled_data_ar,box_new

def get_resampled_with_segs(input,segs,resampled_spacing=[1,1,4],l=True,must_shape=None,newth=0):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler2 = sitk.ResampleImageFilter()
    resampler2.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())

    resampler2.SetOutputSpacing(resampled_spacing)
    resampler2.SetOutputOrigin(input.GetOrigin())
    resampler2.SetOutputDirection(input.GetDirection())
    if must_shape:
        resampler.SetSize(must_shape.GetSize())
        resampler2.SetSize(must_shape.GetSize())
    else:
        resampler.SetSize((1000,1000,1000))
        resampler2.SetSize((1000, 1000, 1000))

    moving_resampled = resampler.Execute(input)

    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)
    if not must_shape:
        try:
            if len(moving_resampled_ar.shape) == 4:
                xx, yy, zz,ee = np.where(moving_resampled_ar > newth)
                moving_resampled_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]
            else:
                xx, yy, zz = np.where(moving_resampled_ar > newth)
                moving_resampled_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
        except:
            a=1

    resampled_data = sitk.GetImageFromArray(moving_resampled_ar)
    #resampled_data.CopyInformation(input)
    #resampled_data.SetSpacing(resampled_spacing)

    mask_map_re = resampler2.Execute(segs)
    mask_map_ar = sitk.GetArrayFromImage(mask_map_re)
    if not must_shape:
        if len(mask_map_ar.shape)==4:
            mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(), 0]
        else:
            mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    mask_map_ar=(mask_map_ar>0)*1.0
    resampled_mask=sitk.GetImageFromArray(mask_map_ar)
    #resampled_mask.CopyInformation(input)
    #resampled_mask.SetSpacing(resampled_spacing)

    return  resampled_data,moving_resampled_ar,resampled_mask,mask_map_ar

def get_resampled_with_segs_and_raws(input,segs,othermods,resampled_spacing=[1,1,4],l=True,must_shape=None):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler2 = sitk.ResampleImageFilter()
    resampler2.SetInterpolator(sitk.sitkNearestNeighbor)

    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())

    resampler2.SetOutputSpacing(resampled_spacing)
    resampler2.SetOutputOrigin(input.GetOrigin())
    resampler2.SetOutputDirection(input.GetDirection())
    if must_shape:
        resampler.SetSize(must_shape.GetSize())
        resampler2.SetSize(must_shape.GetSize())
    else:
        resampler.SetSize((1000,1000,1000))
        resampler2.SetSize((1000, 1000, 1000))

    moving_resampled = resampler.Execute(input)
    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)
    allelse=[]
    for item in othermods:
        item=resampler.Execute(item)
        allelse.append(sitk.GetArrayFromImage(item))
    if not must_shape:
        try:

            if len(moving_resampled_ar.shape) == 4:
                xx, yy, zz,ee = np.where(moving_resampled_ar > 0)
                moving_resampled_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]
            else:
                xx, yy, zz = np.where(moving_resampled_ar > 0)
                moving_resampled_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
            raws=[moving_resampled_ar]
            for item in allelse:
                if len(item.shape) == 4:
                    item_new = item[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]
                else:
                    item_new = item[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
                raws.append(item_new)
        except:
            a=1
    try:
        moving_resampled_ar=np.stack(raws,-1)
        resampled_data = sitk.GetImageFromArray(moving_resampled_ar,isVector=True)
    except:
        a=1
    #resampled_data.CopyInformation(input)
    #resampled_data.SetSpacing(resampled_spacing)

    mask_map_re = resampler2.Execute(segs)
    mask_map_ar = sitk.GetArrayFromImage(mask_map_re)
    if not must_shape:
        if len(mask_map_ar.shape)==4:
            mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(), 0]
        else:
            mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    mask_map_ar=(mask_map_ar>0)*1.0
    resampled_mask=sitk.GetImageFromArray(mask_map_ar)
    #resampled_mask.CopyInformation(input)
    #resampled_mask.SetSpacing(resampled_spacing)

    return  resampled_data,moving_resampled_ar,resampled_mask,mask_map_ar


def get_resampled_with_batches(input,segs,othermods,othersegs,resampled_spacing=[1,1,4],l=True,must_shape=None):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler2 = sitk.ResampleImageFilter()
    resampler2.SetInterpolator(sitk.sitkNearestNeighbor)

    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())

    resampler2.SetOutputSpacing(resampled_spacing)
    resampler2.SetOutputOrigin(input.GetOrigin())
    resampler2.SetOutputDirection(input.GetDirection())
    if must_shape:
        resampler.SetSize(must_shape.GetSize())
        resampler2.SetSize(must_shape.GetSize())
    else:
        resampler.SetSize((1000,1000,1000))
        resampler2.SetSize((1000, 1000, 1000))

    moving_resampled = resampler.Execute(input)
    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)
    allelse=[]
    allsegs=[]
    for item,segs in zip(othermods,othersegs):
        item=resampler.Execute(item)
        segs=resampler2.Execute(segs)
        allelse.append(sitk.GetArrayFromImage(item))
        allsegs.append(sitk.GetArrayFromImage(segs))
    if not must_shape:
        try:

            if len(moving_resampled_ar.shape) == 4:
                xx, yy, zz,ee = np.where(moving_resampled_ar > 0)
                moving_resampled_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]
            else:
                xx, yy, zz = np.where(moving_resampled_ar > 0)
                moving_resampled_ar = moving_resampled_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
            raws=[moving_resampled_ar]
            for item in allelse:
                if len(item.shape) == 4:
                    item_new = item[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(),0]
                else:
                    item_new = item[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
                raws.append(item_new)
        except:
            a=1
    moving_resampled_ar=np.stack(raws,-1)
    resampled_data = sitk.GetImageFromArray(moving_resampled_ar,isVector=True)

    #resampled_data.CopyInformation(input)
    #resampled_data.SetSpacing(resampled_spacing)

    mask_map_re = resampler2.Execute(segs)
    mask_map_ar = sitk.GetArrayFromImage(mask_map_re)
    if not must_shape:
        if len(mask_map_ar.shape)==4:
            mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(), 0]
        else:
            mask_map_ar = mask_map_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
        segs = [mask_map_ar]
        for seg in allsegs:
            if len(seg.shape) == 4:
                seg_new = seg[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max(), 0]
            else:
                seg_new = seg[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
            segs.append(seg_new)
    segs = np.stack(segs, -1)
    #resampled_data = sitk.GetImageFromArray(moving_resampled_ar, isVector=True)
    mask_map_ar=(segs>0)*1.0
    resampled_mask=sitk.GetImageFromArray(mask_map_ar,isVector=True)
    #resampled_mask.CopyInformation(input)
    #resampled_mask.SetSpacing(resampled_spacing)

    return  resampled_data,moving_resampled_ar,resampled_mask,mask_map_ar



def get_matched_segs(input,segs,l=False):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(input.GetSpacing())
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())
    resampler.SetSize(input.GetSize())

    moving_resampled = resampler.Execute(segs)

    moving_resampled_ar = sitk.GetArrayFromImage(moving_resampled)

    resampled_seg = sitk.GetImageFromArray(moving_resampled_ar)
    resampled_seg.CopyInformation(input)
    resampled_seg.SetSpacing(input.GetSpacing())
    if len(moving_resampled_ar.shape)==4:
        moving_resampled_ar=moving_resampled_ar[:,:,:,0]
    return  resampled_seg,moving_resampled_ar
def remap_gray(I,idx,darkmod=0):
    if darkmod==0:
        if idx==1:
            I[I<200]=200
            I[I >2200] = 2200
        elif idx<2:
            I[I<-0]=-0
            I[I >400]= 400
        elif idx==5:
            I[I<-0]=-0
            I[I >750]= 750
        else:
            I[I<-0]=-0
            I[I >600]= 650
    if darkmod==1:
        if idx==1:
            I[I<200]=200
            I[I >2200] = 2200
        elif idx<2:
            I[I<-0]=-0
            I[I >200]= 200
        elif idx==5:
            I[I<-0]=-0
            I[I >350]= 350
        else:
            I[I<-0]=-0
            I[I >300]= 350
    if darkmod==3:
        if idx==1:
            I[I<0]=0
            I[I >3000] = 3000
        elif idx==5:
            I[I < 50] = 50
            I[I >3000]= 3000
        else:
            I[I<50]=50
            I[I >2000]= 2000
    if darkmod==4:
        if idx==1:
            I[I<1150]=1150
            I[I >3200] = 3200
        elif idx==5:
            I[I < 50] = 50
            I[I >3500]= 3500
        else:
            I[I<50]=50
            I[I >2000]= 2000
    if darkmod==2:
        if idx==1:
            I[I<0]=0
            I[I >3000] = 3000
        elif idx==5:
            I[I < -0] = -0
            I[I >2500]= 2500
        else:
            I[I<20]=20
            I[I >2000]= 2000
    I=I-I.min()
    I=I*1.0/I.max()
    return I

def get_margin(segs):
    segs=sitk.ReadImage(segs)
    segs_ar = sitk.GetArrayFromImage(segs)

    kernel = np.ones((3, 3), np.uint8)
    segs_ar_margin=cv2.dilate(segs_ar, kernel, iterations=1) -cv2.erode(segs_ar, kernel, iterations=1)
    segs_margin=sitk.GetImageFromArray(segs_ar_margin)
    segs_margin.CopyInformation(segs)
    return  segs_margin

def get_inside(segs):
    segs=sitk.ReadImage(segs)
    segs_ar = sitk.GetArrayFromImage(segs)
    kernel = np.ones((3, 3), np.uint8)
    segs_ar_margin=cv2.erode(segs_ar, kernel, iterations=2)
    segs_margin=sitk.GetImageFromArray(segs_ar_margin)
    segs_margin.CopyInformation(segs)
    return  segs_margin


def h5viewer(name,moddix,opath):
    f = h5py.File(name, 'r')
    data=f['raw'][moddix,:,:,:]
    ll=(np.sqrt(data.shape[0])//1).astype(np.int)+1
    I=np.zeros((ll*data.shape[1],ll*data.shape[2]))
    J=[]
    for z in range(data.shape[0]):
        J.append(data[z,:,:])
        #I = []
    z=0
    for i in range(ll):
        for j in range(ll):
            I[i*data.shape[1]:(i+1)*data.shape[1],j*data.shape[2]:(j+1)*data.shape[2]]=J[z]
            z+=1
            if z==data.shape[0]:
                break
        if z == data.shape[0]:
            break
    cv2.imwrite(os.path.join(opath,'temp.jpg'),I)


#for item in os.listdir('/mnt/data1/mvi2/h5_croped+mask'):
#    h5viewer(os.path.join('/mnt/data1/mvi2/h5_croped+mask',item),1,'.')