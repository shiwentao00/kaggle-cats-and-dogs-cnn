'''
calculate the mean and std of the images in a given ImageFolder
'''
# import libs
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import argparse
import random
from helper import imshow

def getArgs():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../cat-and-dog/training_set/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-batch_size',
                        type=int,
                        default=256,
                        required=False,
                        help='the batch size, normally 2^n.')

    return parser.parse_args()

def dataSetStatistics(data_dir, batch_size):
    """
    Calculate the statistics of the dataset
    """
    image_size = (256, 256)

    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    #transform = transforms.Compose([transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(data_dir,
                                               transform=transform,
                                               target_transform=None)

    m = dataset.__len__()
    print('length of entire dataset:', m)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=16)

    # calculate mean and std for training data
    mean = 0.
    std = 0.
    # m = 0 # number of samples
    for data,data_label in dataloader:
        # print(data)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1) # reshape
        mean = mean + data.mean(2).sum(0)
        std = std + data.std(2).sum(0)
        # m = m + batch_samples

    mean = mean / m
    std = std / m
    print('mean:',mean)
    print('std:',std)

    return mean, std

def show_example(dataset):
    """
    Randomly show examples of images before normalize
    """    
    m = dataset.__len__()
    index = random.randint(0,m)
    img, _ = dataset.__getitem__(index)
    imshow(img)

if __name__ == "__main__":
    args = getArgs()
    data_dir = args.data_dir
    batch_size = args.batch_size

    image_size = (256, 256)    
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    #transform = transforms.Compose([transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(data_dir,
                                               transform=transform,
                                               target_transform=None)  

    show_example(dataset)
    show_example(dataset)
    show_example(dataset)
    show_example(dataset)

    dataSetStatistics(data_dir, batch_size)