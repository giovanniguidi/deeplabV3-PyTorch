import os
import numpy as np
#import cv2
#from PIL import Image
import random
import datetime
import io
#from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import json
import time

from models.deeplab import *
from data_generators.deepfashion import DeepFashionSegmentation

import argparse
from utils.datagen_utils import denormalize_image
#from deeplab_model.utils.plot_utils import centroid_histogram, mask_and_downsample, get_average_color, normalize_colors
from data_generators.data_generator import initialize_data_loader
from utils.metrics import Evaluator
from tqdm import tqdm
from losses.loss import SegmentationLosses

import yaml
from PIL import Image

class Predictor():
    def __init__(self, config,  checkpoint_path='./snapshots/checkpoint_best.pth.tar'):
        self.config = config
        self.checkpoint_path = checkpoint_path

#        with open(self.config_file_path) as f:

        self.categories_dict = {"background": 0, "short_sleeve_top": 1, "long_sleeve_top": 2, "short_sleeve_outwear": 3,
                "long_sleeve_outwear": 4, "vest": 5, "sling": 6, "shorts": 7, "trousers": 8,
                "skirt": 9,  "short_sleeve_dress": 10, "long_sleeve_dress": 11,
                "vest_dress": 12, "sling_dress": 13}

#        self.categories_dict = {"background": 0, "meningioma": 1, "glioma": 2, "pituitary": 3}
        self.categories_dict_rev = {v: k for k, v in self.categories_dict.items()}
        
        self.model = self.load_model()
        self.train_loader, self.val_loader, self.test_loader, self.nclass = initialize_data_loader(config)

        self.num_classes = self.config['network']['num_classes']
        self.evaluator = Evaluator(self.num_classes)
        self.criterion = SegmentationLosses(weight=None, cuda=self.config['network']['use_cuda']).build_loss(mode=self.config['training']['loss_type'])


    def load_model(self):
        model = DeepLab(num_classes=self.config['network']['num_classes'], backbone=self.config['network']['backbone'],
                        output_stride=self.config['image']['out_stride'], sync_bn=False, freeze_bn=True)


        if self.config['network']['use_cuda']:
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location={'cuda:0': 'cpu'})

#        print(checkpoint)
        model = torch.nn.DataParallel(model)

        model.load_state_dict(checkpoint['state_dict'])

        return model

    def inference_on_test_set(self):
        print("inference on test set")

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.config['network']['use_cuda']:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print("Accuracy:{}, Accuracy per class:{}, mean IoU:{}, frequency weighted IoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)



    def segment_image(self, filename):

#        file_path = os.path.join(dir_path, filename)
        img = Image.open(filename).convert('RGB')

        sample = {'image': img, 'label': img}

        sample = DeepFashionSegmentation.preprocess(sample, crop_size=513)
        image, _ = sample['image'], sample['label']
        image = image.unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(image)

        image = image.squeeze(0).numpy()
        image = denormalize_image(np.transpose(image, (1, 2, 0)))
        image *= 255.

        prediction = prediction.squeeze(0).cpu().numpy()

#        print(prediction[])

        prediction = np.argmax(prediction, axis=0)

        return image, prediction