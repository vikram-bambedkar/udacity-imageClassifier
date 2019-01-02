#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:19:39 2018

@author: vikram
"""
import argparse
import torch
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image

# Setting up command line arguments

parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store',
                    help='Specify image path.')

parser.add_argument('checkpoint', action='store',
                    help='Specify checkpoint path.')

parser.add_argument('--category_names', action='store',
                    default='cat_to_name.json',
                    dest='category_names',
                    help='Specify mapping of categories to real names')

parser.add_argument('--top_k', action='store',
                    default=1,
                    dest='top_k',
                    type=int, 
                    help='Return top K most likely classes. Default is 3.')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set mode to GPU. Default is FALSE.')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

arguments = parser.parse_args()

# Checking if GPU processing has been selected
device = torch.device("cuda:0" if arguments.gpu else "cpu")

# ### Label mapping

with open(arguments.category_names, 'r') as f:
    cat_to_name = json.load(f)
    

# ## Loading the checkpoint
def load_checkpoint(path):
    
    checkpt = torch.load(path)
    epochs = checkpt['epochs']
    arch = checkpt['arch']

    if arch == 'vgg':
        model = models.vgg11(pretrained=True)
    else:
        model = models.densenet201(pretrained=True)


    model.classifier = checkpt['classifier']
    model.class_to_idx = checkpt['class_idx']

    model.load_state_dict(checkpt['state_dict'])
    
    return model, epochs


# Load the checkpoint
model, epochs = load_checkpoint(arguments.checkpoint)

# # Inference for classification

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    im = Image.open(image)
    im = im.resize(size)
    im = im.crop(box=(16,16,240,240))
    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = np.array(im)/255
    
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2,0,1))
    
    return np_image


# imshow function to test image processing and displaying the image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction

# Function to plot the image and bar chart of probabilities
def plot_barh(probability, cls_name, img_to_show):
    
    #imshow(img_to_show)
    
    prob = probability.detach().numpy()

    dataframe = pd.DataFrame(prob.T)
    dataframe[1] = cls_name

    fig, ax = plt.subplots(2,1)
    imshow(img_to_show,ax[0])
    ax[1].barh(dataframe.index, dataframe[0], align='center', tick_label=dataframe[1])
    ax[1].invert_yaxis()
    fig.suptitle(dataframe[1][0])
    plt.show()


model.to(device)
# Function to predict the image
def predict(image_path, model, topk=arguments.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    pp_img = process_image(image_path)
    
    pp_img = pp_img[None,:,:,:]
    
    img_tensor = torch.from_numpy(pp_img)
    img_tensor = img_tensor.to(device)
    if arguments.gpu:
        img_tensor = img_tensor.type(torch.cuda.FloatTensor)
    else:
        img_tensor = img_tensor.type(torch.FloatTensor)
    
    
    op = model.forward(img_tensor)
    
    prob, classes = torch.topk(op,topk)
    prob = prob.exp()
    
    return prob, classes


# Function to get Flower name
def get_cat_name(cls_idx_dict, idx_arr, cat_to_name):
    cls_name = []
    for x in np.nditer(idx_arr):
        for cls, idx in cls_idx_dict.items():
            if x == idx:
                cls_name.append(cat_to_name[cls])
    
    return np.array(cls_name)


# Function to Display an image along with the top 5 classes
def check_pred(path, model):
    cls_idx = model.class_to_idx
    probability, classes = predict(path,model)
    
    cls_name = get_cat_name(cls_idx, classes, cat_to_name)
    #img_to_show = process_image(path)
    
    #plot_barh(probability, cls_name, img_to_show)
    
    print('Class Name(s)       = {!r}'.format(cls_name))
    print('Probability         = {!r}'.format(probability))
    

# Call to check_pred function
check_pred(arguments.image_path ,model)
