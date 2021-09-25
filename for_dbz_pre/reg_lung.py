import SimpleITK as sitk
import numpy as np
import os,glob,cv2,h5py,random
from utils_sitk import *
#from multiprocessing import Pool as threadpool
raw_path='/mnt/newdisk3/data_for2d'
seg_path='/mnt/newdisk3/seg_for2d'
atlas='/mnt/newdisk2/reg/atlas/dataatlas.nrrd'
seg='/mnt/newdisk2/reg/atlas/segatlas.nrrd'
lung='/mnt/newdisk2/reg/atlas/lung.nrrd'
output_path='/mnt/newdisk2/reg/new_t'
os.makedirs(output_path,exist_ok=True)
#os.makedirs(output_path2,exist_ok=True)
type1=os.listdir(raw_path)

atlas=sitk.ReadImage(atlas)
#atlas.SetOrigin((0,0,0))
lung=sitk.Cast(sitk.ReadImage(lung),sitk.sitkFloat32)
#lung.SetOrigin((0,0,0))type1
seg=sitk.ReadImage(seg)
#seg.SetOrigin((0,0,0))type1
seg_ar=sitk.GetArrayFromImage(seg).astype(np.float32)
Maps=[]
for i in range(seg_ar.shape[-1]):
    t=sitk.GetImageFromArray(seg_ar[:,:,:,i])
    t.CopyInformation(seg)
    Maps.append(t)


R = sitk.ImageRegistrationMethod()
R.SetMetricAsJointHistogramMutualInformation(20)

R.SetMetricSamplingPercentage(0.01)
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetShrinkFactorsPerLevel([3, 2, 1])
R.SetSmoothingSigmasPerLevel([2,1, 1])
R.MetricUseFixedImageGradientFilterOff()
R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                numberOfIterations=100,
                                estimateLearningRate=R.EachIteration)
R.SetOptimizerScalesFromPhysicalShift()
R.SetInterpolator(sitk.sitkLinear)
#reliable=dict()
#random.shuffle(type1)
for item in type1:
    type2=os.listdir(os.path.join(raw_path,item))
    #random.shuffle(type2)
    for item2 in type2:
        allpatient=os.listdir(os.path.join(raw_path, item,item2))
        random.shuffle(allpatient)
        for patient in allpatient:
            if os.path.exists(os.path.join(output_path,item,item2,patient.replace('.nii','.seg.nii'))):
                continue
            try:
                data = sitk.ReadImage(os.path.join(raw_path,item,item2,patient))
            except:
                continue
            #ro=data.GetOrigin()
            #data.SetOrigin((0,0,0))
            if not os.path.exists(os.path.join(seg_path, item, item2, patient)):
                continue
            segs = sitk.ReadImage(os.path.join(seg_path, item, item2, patient))
            segs=sitk.Cast(segs,sitk.sitkFloat32)
            #segs.SetOrigin((0,0,0))

            try:
                R.SetInitialTransform(
                    sitk.CenteredTransformInitializer(
                        segs,
                        lung,
                        sitk.AffineTransform(3),
                        sitk.CenteredTransformInitializerFilter.MOMENTS,
                    )
                )
                outTx = R.Execute(segs, lung)
            except:
                continue
            interpolator = sitk.sitkNearestNeighbor
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolator)
            resampler.SetDefaultPixelValue(0)
            resampler.SetReferenceImage(segs)
            resampler.SetTransform(outTx)
            data_ar=sitk.GetArrayFromImage(segs)
            results=np.zeros_like(data_ar).astype(np.uint8)
            #trans_atlas = sitk.Transformix(atlas, elastixImageFilter.GetTransformParameterMap())
            try:
                for i in range(len(Maps)):
                    resultLabel=resampler.Execute(Maps[i])
                    #resultLabel = sitk.Transformix(Maps[i], elastixImageFilter.GetTransformParameterMap())
                    resultLabel=sitk.GetArrayFromImage(resultLabel)
                    resultLabel=(resultLabel>0.5).astype(np.uint8)
                    results[resultLabel==1]=i+1
            except:
                continue
            #resultLabel=np.stack(results,-1)
            resultLabel=sitk.GetImageFromArray(results)
            resultLabel.CopyInformation(segs)
            moved=resampler.Execute(segs)

            interpolator = sitk.sitkBSpline
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolator)
            resampler.SetDefaultPixelValue(0)
            resampler.SetReferenceImage(segs)
            resampler.SetTransform(outTx)
           
            move_raw=resampler.Execute(data)

            os.makedirs(os.path.join(output_path,item,item2),exist_ok=True)
            sitk.WriteImage(resultLabel,os.path.join(output_path,item,item2,patient.replace('.nii','.seg.nii')))
            #sitk.WriteImage(data, os.path.join(output_path, item, item2, 'data.nii'))
            #sitk.WriteImage(moved, os.path.join(output_path, item, item2, patient.replace('.nii','.moved.nii')))
            print('OK!',item,item2,patient)
    print(type2,'OK!')
#pools = threadpool(6)
#pools.map(pro, type1)
#pools.close()
#pools.join()






