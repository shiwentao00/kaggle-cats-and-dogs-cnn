
"""
 Bionois hydroph classifier using transfer learning
 A classifier is built to distinguish 'control' vs 'heme' or 'control' vs 'nucleotide'.
 A pre-trained CNN(resnet18) is used as the feature extractor. New fully-connected layers are
 stacked upon the feature extractor, and we fine-tune the entire Neural-Network since the orginal
 CNN is trained on a different data domain(imageNet)
 """
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import time
import copy
import sklearn.metrics as metrics
from helper import imshow
from utils import dataPreprocess
from utils import cats_vs_dogs_config
from utils import train_model

def get_args():
    """
    The parser function to set the default vaules of the program parameters or read the new value set by user.
    """    
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='cat',
                        required=False,
                        choices = ['cat', 'dog', 'cat_vs_dog'],
                        help='operation mode, cat or dog or cat_vs_dog.')

    parser.add_argument('-seed',
                        default=123,
                        required=False,
                        help='seed for random number generation.')

    parser.add_argument('-data_dir',
                        default='../data_prep/cats_vs_dogs/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-model_dir',
                        default='./model/cats_vs_dogs_resnet18.pt',
                        required=False,
                        help='file to save the model for cats vs dogs.')

    parser.add_argument('-batch_size',
                        type=int,
                        default=256,
                        required=False,
                        help='the batch size, normally 2^n.')

    parser.add_argument('-normalizeDataset',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')
    return parser.parse_args()

# get input parameters for the program
args = get_args()
op = args.op
seed = args.seed
data_dir = args.data_dir
model_dir = args.model_dir
batch_size = args.batch_size
print('batch size: '+str(batch_size))
normalizeDataset = args.normalizeDataset

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device: '+str(device))

# put data into 3 DataLoaders
trainloader, valloader = dataPreprocess(op, batch_size, data_dir, seed, normalizeDataset)

# create the data loader dictionary
dataloaders_dict = {"train": trainloader, "val": valloader}

'''
Train the Neural Network
'''
# import the model and training configurations
net, loss_fn, optimizerDict, num_epochs = cats_vs_dogs_config(device)

# train the neural network
trained_model, best_mcc_roc, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history = train_model(net,
                                                 device,
                                                 dataloaders_dict,
                                                 loss_fn,
                                                 optimizerDict,
                                                 num_epochs = num_epochs
                                                 )
# The Plottings
# plot train/validation loss vs # of epochs
fig1 = plt.figure()
plt.plot(train_loss_history,label='train')
plt.plot(val_loss_history,label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Train Loss vs Validation Loss")
plt.legend()
plt.draw()

# plot train/validation accuracy vs # of epochs
fig2 = plt.figure()
plt.plot(train_acc_history,label='train')
plt.plot(val_acc_history,label='validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title("Train Accuracy vs Validation Accuracy")
plt.legend()
plt.draw()

# plot train/validation mcc vs # of epochs
fig3 = plt.figure()
plt.plot(train_mcc_history,label='train')
plt.plot(val_mcc_history,label='validation')
plt.xlabel('epoch')
plt.ylabel('MCC')
plt.title("Train MCC vs Validation MCC")
plt.legend()
plt.draw()

# save the trained model with best MCC value
torch.save(trained_model.state_dict(), model_dir)

# show all the figures
plt.show()

print('training finished, end of program')