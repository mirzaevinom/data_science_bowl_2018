# The 2018 Data Science Bowl: "Spot Nuclei. Speed Cures."

This repository contains scripts of my solution to [The 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). Goal of the competition was to create an algorithm to automate nucleus detection from biomedical images.

This model is based on [Matterport](https://github.com/matterport/Mask_RCNN)'s Mask-RCNN implementation. Mask-RCNN has been successfully used to [find features in satellite images](https://github.com/jremillard/images-to-osm) and detect objects in [autonomous driving](https://www.youtube.com/watch?v=OOT3UIXZztE&ab_channel=KarolMajek). I found [this tutorial](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) very useful for understanding the model.

Here are the main changes that I have made to the original model:
* Original Matterport model was only validating on only one image so fixed this [validation issue](https://github.com/matterport/Mask_RCNN/issues/89).
* Training on mosaics from train images based on [Emil](https://www.kaggle.com/bonlime/train-test-image-mosaic)'s notebook and [Nikolay Shebanov](https://github.com/killthekitten/kaggle-ds-bowl-2018-baseline/blob/master/rebuild_mosaics.py)'s codes.
* Using [external data](https://www.kaggle.com/voglinio/external-h-e-data-with-mask-annotations) for training. Splitting images into four pieces due to large size. External dataset [download links](https://nucleisegmentationbenchmark.weebly.com/dataset.html).
* Using stratification for splitting the dataset into train and validation (using a csv file provided by [Allen](https://www.kaggle.com/c/data-science-bowl-2018/discussion/48130)).
* Added extra parameter `DETECTION_MASK_THRESHOLD` to model configuration. Default is 0.5 but I achieved better results with 0.35.
* Added extra parameter `OPTIMIZER` to model configuration. Default is `SGD` but I found that model reaches optimimum much faster with `ADAM`.
* Added a lot of image augmentation functions:
  * Random vertical flips
  * Random 90 or -90 degrees rotation
  * Modified and added [Heng CherKeng](https://www.kaggle.com/c/data-science-bowl-2018/discussion/49692)'s PyTorch functions for random image cropping (bigger images to 256x256x3), scaling (from 0.5 to 2.0) and rotating (from -15 to 15 degrees).
  * Also implemented random Gaussian blur but it did not improve model performance.


* Combining predictions on actual image and LR flipped image to get slightly better accuracy.
* Removing overlaps based on objectness scores. In other words, removing overlapped regions from masks with lower scores.
  * If this intersection removal results in multiple objects, then removing all the small pieces
* Dilating and then eroding individual masks helped me to achieve slightly better result.
* If a model predicts no masks for an image then I rescale the image and predict once again.

### Some model predictions
For the following figures red lines represent ground truth boundaries and blue lines represent prediction boundaries.

* Model predictions for some stage 1 test image samples (public LB in stage 1: __0.502__)

<img src="images/sample_1.png" height="300" width="30%"/> <img src="images/sample_2.png" height="300" width="30%"/> <img src="images/sample_3.png" height="300" width="30%"/>

* Model predictions for some stage 2 test image samples (private LB in stage 2: )

<img src="images/sample_4.png" height="300" width="30%"/> <img src="images/sample_5.png" height="300" width="30%"/> <img src="images/sample_6.png" height="300" width="30%"/>


### Dependencies

The codes are written and tested on Python >=3.5. These scripts depend on the following libraries:
* `tensorflow>=1.3, keras>=2.1.3, numpy, scipy` for computations
* `cv2, skimage, matplotlib` for image processing and plotting
* `tqdm` for progress bar

### Usage

1. Run `python augment_preprocess.py` to pre-process external data and create mosaics from the dataset. (You can skip this step if you only want to train on provided train set)

2. Run `python train.py` to train the model. Model weights are saved at `../data/logs/kaggle_bowl/mask_rcnn.h5`.

3.  Run `python predict.py` to evaluate model performance on validation set and predict nuclei boundaries on test set.

### Folder structure

```
- codes
 |
- data
 |
 ---- stage1_train
 |    |
 |    ---- imageID
 |    |    |
 |    |    ---- images
 |    |    |
 |    |    ---- masks
 ---- stage1_test
 |    |
 |    ---- similar to stage1_train
 ---- stage2_test
 |    |
 |    ---- similar to stage1_train
 ---- external_data
 |    |
 |    ---- tissue_images
 |    |    |
 |    |    ---- imageID.tif
 |    |    
 |    ---- annotations
 |    |    |
 |    |    ---- imageID.xml
```

### Acknowledgements

* This material is based upon work supported by the National Science Foundation under Agreement No. 0931642 ([The Ohio Supercomputer Center](https://www.osc.edu/))
* I would like to also thank [Mathematical Biosciences Institue](http://mbi.osu.edu) (MBI) at Ohio State University, for partially supporting this research. MBI receives its funding through the National Science Foundation grant DMS 1440386
