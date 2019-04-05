'''
calculate the mean and std of the images in a given ImageFolder
'''
# import libs
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import copy
import argparse

def getArgs():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-opMode',
                        default='control_vs_heme',
                        required=False,
                        help='Operation mode, control_vs_heme, or control_vs_nucleotide')

    parser.add_argument('-dataDir_control_vs_heme',
                        default='../../../../../../work/wshi6/deep-learning-data/control_vs_heme/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-dataDir_control_vs_nucleotide',
                        default='../../../../../../work/wshi6/deep-learning-data/control_vs_nucleotide/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-batchSize',
                        type=int,
                        default=256,
                        required=False,
                        help='the batch size, normally 2^n.')

    return parser.parse_args()

def dataSetStatistics(dataDir, batchSize):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(dataDir,
                                               transform=transform,
                                               target_transform=None)

    m = dataset.__len__()
    print('length of entire dataset:', m)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=False,
                                             num_workers=1)

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

if __name__ == "__main__":
    args = getArgs()
    opMode = args.opMode
    if opMode == 'control_vs_heme':
        dataDir = args.dataDir_control_vs_heme
    elif opMode == 'control_vs_nucleotide':
        dataDir = args.dataDir_control_vs_nucleotide
    batchSize = args.batchSize

    # use training set statistics to approximate the statistics of entire dataset
    dataDir = dataDir + '/train'
    dataSetStatistics(dataDir, batchSize)