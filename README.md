# DeepLab V3+ Network for Semantic Segmentation

This project is an application of a segmenting segmentation algorithm to the DeepFashion2 dataset 
(Ge et al. 2019), containing images of 13 popular clothing categories from both commercial shopping stores and consumers.


The task this class of algorithms want to 
solve is "semantic segmentation", i.e. given a picture assign each pixel a "semantic" label, such as tree, street, sky, car. 


We use the DeepLab V3+ Network for Semantic Segmentation model (Chen et al. 2018, https://arxiv.org/abs/1802.02611), one of the state-of-the-art models
in semantic segmentation 


![picture alt](https://github.com/giovanniguidi/FCN-keras/blob/master/figures/semantic_segmentation.jpg "")

## Virtual environment
First you need to create a virtual environment (using Conda for instance) by:
```
conda create --name deeplab --python==3.7.1
conda activate deeplab
```


## Depencencies

Install the libraries using:
```
pip install -r requirements.txt 
```

## Data


To download the dataset go to: 

https://github.com/switchablenorms/DeepFashion2

and download all the images.

To convert the labels in ones useful for semantic segmentation (in this case .png images, where the value of each pixel is the cloth class), you need
to convert the segmentation points into the proper format. This has already been done, so you can download the folder from


https://drive.google.com/drive/folders/1O8KLZa1AABlLS6DlkkzHOgPqvT89GB_9?usp=sharing

which also contain the train/val/test split


![picture alt](https://github.com/giovanniguidi/FCN-keras/blob/master/test_images/2010_001403.jpg "")



## Weights

The trained weights can be found at:

https://drive.google.com/open?id=1JXfM5X0aihv2d_4WN8_bIvzrfhB0Me5k


You can train the model with different backbones (resnet, xception, drn, mobilenet), this model has been trained with resnet backbone


## Train

To train a model run:

```
python main.py -c configs/config.yml --train
```

If you set "weights_initialization" in config.yml you can use a pretrained model to inizialize the weights, usually for restoring the training after an interruption.  

During training the best and last snapshots can be stored if you set those options in "callbacks" in config.yml.


## Inference 

To predict on the full test set run: 

```
python main.py -c configs/config.yml --predict_on_test
```


To predict on a single image you can run:

```
python main.py -c configs/config.yml --predict --filename test_images/test_images/2010_004856.jpg
```

In "./test_images/" there are some images that can be used for testing the model. 


## Results

Here is an example of prediction:

![picture alt](https://github.com/giovanniguidi/FCN-keras/blob/master/figures/pred_1.jpg "")

Check "inference.ipynb" in notebooks for a visual assessment of the prediction.

On the test set we get this metrics (see https://arxiv.org/pdf/1411.4038.pdf for the definition):

```
pixel accuracy: 0.81
mean accuracy: 0.35
mean IoU: 0.27
freq weighted mean IoU: 0.69
````

## Train on other data

This implementation can be easily extended to other dataset. The expected input are .jpg images, and the labels must be in .png format, with 1 channel shape (y_size, x_size) and value corresponding to the desired class
 

## Technical details

Images are normalized between -1 and 1.

Data can be augmented by flipping, translating, and changing random_brightness and random_saturation.


## To do

- [x] 


## References


\[1\] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
\[2\] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
