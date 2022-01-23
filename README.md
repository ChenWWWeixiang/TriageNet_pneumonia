# A deep learning pathogen recognition system for pneumonia based on CT

This is the code for pathogen recorgnition of pneumonia. The base of this work forked our former work https://github.com/ChenWWWeixiang/diagnosis_covid19.

Guidance to Use
---------------

### Environment

run ``pip install -r requirements.txt`` to install all above packages.

### Usage

```
python testengine.py -p <path to trainedmodel> -m <list of paths for lung segmentation> -i <list of paths for image data> -o <path to save record> -g <gpuid>
```

### Train on Your Own Data

1. __Data Preparation__ : The datasets from Wuhan Union Hospital, Western Campus of Wuhan Union Hospital, and Jianghan Mobile Cabin Hospital were used under the license of the current study and are not publicly available. Applications for access to the LIDC-IDRI database can be made at [https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). ILD-HUG database can be accessed at [http://medgift.hevs.ch/wordpress/databases/ild-database/](http://medgift.hevs.ch/wordpress/databases/ild-database/).
2. __Volumes to Images__: We suggest that test data should be in ".nii" format (any formats that *SimpleITK* can work on is OK with small changes in codes) and training data should be in ".jpg" format (any formats that *opencv-python* can work on is OK with small changes in codes). A script "data/test_hu.py" is used to cut volumes into images.
3. __Lung Segmentation__ : using Deeplabv1 (https://github.com/DrSleep/tensorflow-deeplab-resnet)
   or any other segmentation methods.
4. __Split Dataset__:

```
python data/get_set_seperate_jpg.py -p <list of paths to jpgs for seperate> -t <train list output path> -v <validation list output path>
```

5. __Begin Training__: training parameters are listed on ``options_stack.toml``. Run ``python main.py `` to train the model.
6. __Metrics__: to evaluate the performances, run ``python testengine.py -p <path to trainedmodel> -m <list of paths for lung segmentation> -i <list of paths for image data> -o <path to save record> -g <gpuid>``
   and then the script ``python ploc_roc.py -i <list of paths for recording npys> -o <path to save metrics>``

### More Research Tools

* __Model Visualization__:
  A script to show Grad-CAM/CAM result is available. Input images should be in jpg formats and should be concatenated with lung mask as Red channel. The input raw jpgs and input masked jpgs should be in pair:

```
python models/gradcam.py --image_path <raw jpg img path> --mask_path <jpg img with mask path> --model_path <path to trained model> --output_path <path to output>
```

You can also use our volume cam script, which demands   raw data volumes and lung segmentation volumes as inputs.

```
python models/grad_volume.py --image_path <raw data nii path> --mask_path <lung mask nii file path> --model_path <path to trained model> --output_path <path to output>
```

* __Fractal Dimension Features__ :

  - __Extract Fractal Dimension__: extract fractal dimension of a region.``python fractal-dimension/fractals.py -i <binary nii file determined regions> -o <output txt path>``
  - __Extract 3D mesh Fractal Dimension__: extract fractal dimension of gray level mesh. The input data and input region should be in pair.``python fractal-dimension/fractal.py -m <binary nii file determined regions> -r <nii file of raw data> -o <output txt path> ``
* __Extract Radiomics Features__ : parameters of radiomics are listed in ``radiomics/RadiomicsParams.yaml``. Run ``python get_r_features.py m <binary nii file determined regions> -r <nii file of raw data> -o <output csv path>`` to get radiomics features of a region. The input data and input region should be in pair.
* __LASSO Analysis__ : this script analysis radiomics features using LASSO. ``python plot_lasso_mse.py -i <input csv file >``
* __Abnormal Locating__ : we fine-tune the trained model in slices from only COVID-19 positives in order to train a model to locate abnormal slices in COVID-19 positive volumes. Test the model using ``python multi_period_scores/get_abnormal_scores.py`` and visualize the results using ``python analysis_mp.py``. Some of our patients have multi-period CTs, the visualization of abnormal slices shows the changes of lesion with time.

Citation
--------

If you find this project helpful, please cite our paper:

```
@article{chen2021deep,
  title={Deep diagnostic agent forest (DDAF): A deep learning pathogen recognition system for pneumonia based on CT},
  author={Chen, Weixiang and Han, Xiaoyu and Wang, Jian and Cao, Yukun and Jia, Xi and Zheng, Yuting and Zhou, Jie and Zeng, Wenjuan and Wang, Lin and Shi, Heshui and others},
  journal={Computers in biology and medicine},
  pages={105143},
  year={2021},
  publisher={Elsevier}
}
```
