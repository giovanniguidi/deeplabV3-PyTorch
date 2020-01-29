# DeepLab V3+ Network for Semantic Segmentation

This project is based on one of the state-of-the-art algorithms for semantic segmentation, DeepLabV3+ by the Google research group (Chen et al. 2018, https://arxiv.org/abs/1802.02611). Semantic segmentation is the task of predicting for each pixel of an image a "semantic" label, such as tree, street, sky, car (and of course background). 


This algorithm is here applied to the DeepFashion2 dataset (Ge et al. 2019), one of the most popular dataset used by fashion research groups. The 
dataset contains 491K images of 13 popular clothing categories with bounding boxes, and almost 185K images with segmentation, from both commercial shopping stores and consumers.

DeepLabV3+ model is very complex, but the biggest difference compared to other models is the use of "atrous convolutions" in the encoder (which was already suggested in the first DeepLab model by Chen et al. 2016), in a configuration called Atrous Spatial Pyramid Pooling (ASPP). ASPP is composed by different atrous convolution layers in parallel with a different atrous rate, allowing to capture information at multiple scales and extract denser 
feature maps (see the image below and the paper for details).  

<p align="center">
  <img src="https://github.com/giovanniguidi/deeplabV3_Pytorch/blob/master/docs/deeplab.png">
  <i>Fig. 1: DeepLabV3+ model (source Chen et al. 2018)</i>
</p>


## Virtual environment
First you need to create a virtual environment. 

Using Conda you can type:

```
conda create --name deeplab --python==3.7.1
conda activate deeplab
```


## Dependencies
This project is based on the PyTorch Deep Learning library. 

Install the dependencies by:
```
pip install -r requirements.txt 
```

## Dataset

Download the dataset from: 

https://github.com/switchablenorms/DeepFashion2


Before using those data you need to convert the labels in a format which can seamless enter into a semantic segmentation algorithm. In this case we use .png images, where the value of each pixel is the cloth class (so from 1 to 13), but other choices are possible. The background class has value 0.

You need to create a script to convert the polygons in DeepFashion2 labels into a proper format for the algorithm, or you can download the labels from:

https://drive.google.com/drive/folders/1O8KLZa1AABlLS6DlkkzHOgPqvT89GB_9?usp=sharing


This folder contains also the train/val/test split json in case you want to use the same split I used.

## Parameters

All the parameters of the model are in configs/config.yml.

## Weights

The trained weights can be found here:

https://drive.google.com/drive/folders/1O8KLZa1AABlLS6DlkkzHOgPqvT89GB_9?usp=sharing


The model can be trained with different backbones (resnet, xception, drn, mobilenet). The weights on the Drive has been trained with the ResNet backbone, so if you want to use another backbone you need to train from scratch (although the backbone weights are always pre-trained on ImageNet).


## Train

To train a model run:

```
python main.py -c configs/config.yml --train
```

You can set "weights_initialization" to "true" in config.yml, in order to restore the training after an interruption.  

During training the best and last snapshots can be stored if you set those options in "training" in config.yml.


## Inference 

To predict on the full test set run and get the metrics do: 

```
python main.py -c configs/config.yml --predict_on_test
```

In "./test_images/" there are some images that can be used for testing the model. To predict on a single image you can run:

```
python main.py -c configs/config.yml --predict --filename test_images/068834.jpg
```

You can also check the "inference.ipynb" notebook for visual assessing the predictions.


## Results

Here is an example of the results:

<p align="center">
  <img src="https://github.com/giovanniguidi/deeplabV3_Pytorch/blob/master/docs/sample.png">
  <i>Fig. 2: Prediction on DeepFashion2</i>
</p>

On the test set we get this metrics:

```
accuracy: 0.84
accuracy per class: 0.47
mean IoU: 0.34
freq weighted IoU: 0.79
````


## Train on other data

This implementation can be easily used on other dataset. The expected input of the model are .jpg images, and the labels are in .png format, with 1 channel (i.e. shape = (y_size, x_size)), and pixel value corresponding to the target class. In principle you only need to modify the data_generator.  
 

## References


\[1\] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

\[2\] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
