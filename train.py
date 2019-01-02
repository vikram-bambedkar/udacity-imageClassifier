#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports here
import torch
import argparse
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict

# %matplotlib inline

# Setting up command line arguments

parser = argparse.ArgumentParser()

parser.add_argument('data_directory', action='store',
                    help='Store a Data Directory location')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='Specify a directory to save checkpoints')

parser.add_argument('--arch', action='store',
                    default='vgg',
                    dest='arch',
                    help='Specify architecture. Select from vgg(default) or densenet.')

parser.add_argument('--learning_rate', action='store',
                    default=0.001,
                    dest='learning_rate',
                    type=float, 
                    help='Specify learning rate. Default is 0.001.')

parser.add_argument('--hidden_units', action='store',
                    default=512,
                    dest='hidden_units',
                    type=int, 
                    help='Specify hidden units. Default is 512.')

parser.add_argument('--epochs', action='store',
                    default=3,
                    dest='epochs',
                    type=int, 
                    help='Specify epochs. Default is 3.')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set mode to GPU. Default is FALSE.')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

arguments = parser.parse_args()

# Checking if GPU processing has been selected
device = torch.device("cuda:0" if arguments.gpu else "cpu")

# ## Load the data
data_dir = arguments.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                       transforms.RandomResizedCrop(224), 
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406],  
                                                            [0.229, 0.224, 0.225] )
                                      ])

valid_transforms = transforms.Compose([transforms.Resize(256), 
                                       transforms.CenterCrop(224), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406],  
                                                            [0.229, 0.224, 0.225] )
                                      ])

test_transforms = transforms.Compose([transforms.Resize(256), 
                                       transforms.CenterCrop(224), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406],  
                                                            [0.229, 0.224, 0.225] )
                                      ])

# Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms) 
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)




# # Building and training the classifier

# Build and train your network. Consider adding options to select an algorithm
def build_model(modelName, hidden_units):
    if modelName == 'vgg':
        model = models.vgg11(pretrained=True) #instantiate the model
        
        for param in model.parameters(): 
            param.requires_grad = False
        #Replace the classifier
        classifier_fl = nn.Sequential(OrderedDict([
            ('cfl1', nn.Linear(25088,hidden_units)), 
            ('relu1', nn.ReLU()), 
            ('drp1', nn.Dropout(0.3)), 
            ('cfl3', nn.Linear(hidden_units, 102)),
            ('out', nn.LogSoftmax(dim=1))
            ]))

        model.classifier = classifier_fl
        arch = 'vgg'

    elif modelName == 'densenet':
        model = models.densenet201(pretrained=True) #instantiate the model
        
        for param in model.parameters(): 
            param.requires_grad = False
        #Replace the classifier
        classifier_fl = nn.Sequential(OrderedDict([
            ('cfl1', nn.Linear(1920,hidden_units)), 
            ('relu1', nn.ReLU()), 
            ('drp1', nn.Dropout(0.3)), 
            ('cfl3', nn.Linear(hidden_units, 102)),
            ('out', nn.LogSoftmax(dim=1))
            ]))

        model.classifier = classifier_fl
        arch = 'densenet'
        
    else:
        model = models.vgg11(pretrained=True)
        
        for param in model.parameters(): 
            param.requires_grad = False
        #Replace the classifier
        classifier_fl = nn.Sequential(OrderedDict([
            ('cfl1', nn.Linear(25088,hidden_units)), 
            ('relu1', nn.ReLU()), 
            ('drp1', nn.Dropout(0.3)), 
            ('cfl3', nn.Linear(hidden_units, 102)),
            ('out', nn.LogSoftmax(dim=1))
            ]))

        model.classifier = classifier_fl
        arch = 'vgg'
        
    return model, arch

model, arch = build_model(arguments.arch ,arguments.hidden_units)

# Define Loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)

#function to validate while training

def validate(model, valloader, criterion):
    #setting validation loss and accuracy as zero
    val_loss = 0
    accuracy = 0
    #iterating through validation loader to feed forward and to track the error/accuracy
    for img, lbl in valloader:
        img, lbl = img.to(device), lbl.to(device)
        op = model.forward(img)
        val_loss += criterion(op, lbl).item()
        ps = torch.exp(op)
        eq = (lbl.data == ps.max(dim=1)[1])
        accuracy += eq.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy

# Training the network. Convert this to a function
epochs = arguments.epochs
print_every = 40
steps = 0

#model to cuda
model.to(device)

#iterating for given number of epochs
for e in range(epochs):
    model.train()
    running_loss = 0
    #iterating through the training data
    for i, (image, label) in enumerate(trainloader):
        steps += 1
        
        #image, label to cuda
        image, label = image.to(device), label.to(device)
        #setting gradient as zero
        optimizer.zero_grad()
        
        #Forward pass and backpropagation
        op = model.forward(image)
        loss = criterion(op, label)
        loss.backward() #calculates the gradient
        optimizer.step() #updating the weights
        
        running_loss += loss.item()
        
        #printing stats
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss, accuracy = validate(model, validloader, criterion)
            
            print("Epoch: {}/{}   ".format(e+1, epochs), 
            "Training Loss: {:.4f}   ".format(running_loss/print_every), 
            "Validation Loss: {:.4f}   ".format(val_loss/len(validloader)), 
            "Accuracy: {:.2f}".format(100*accuracy/len(validloader)))
            
            running_loss = 0
            model.train()


# ## Testing your network

# Validation on the test set
def chk_test_accuracy(testloader):
    correct = 0
    total = 0
    model.to('cpu')
    with torch.no_grad():
        for img_test, lbl_test in testloader:
            op_test = model(img_test)
            _, pred_test = torch.max(op_test.data, 1)
            eq_test = (pred_test == lbl_test)
            
            correct += eq_test.type(torch.FloatTensor).sum().item()
            total += lbl_test.size(0)
    return correct, total
        
# Calling the function
pred_true, total = chk_test_accuracy(testloader)
print("Accuracy for Test data: {:.2f} %".format(100* pred_true/total))


# ## Save the checkpoint
checkpoint = {
    'epochs': arguments.epochs,
    'classifier': model.classifier, 
    'state_dict': model.state_dict(), 
    'class_idx': train_datasets.class_to_idx,
    'arch': arch
}

torch.save(checkpoint, arguments.save_dir)
