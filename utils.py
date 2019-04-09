"""
Modules and functions to build the CNN
"""
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import copy
import time
import sklearn.metrics as metrics
from torchvision import datasets, models, transforms
from math import sqrt
from torch.autograd import Variable
from helper import calc_metrics

def getArgs():
    """
    The parser function to set the default vaules of the program parameters or read the new value set by user.
    """    
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-op',
                        default='cat',
                        required=False,
                        choices = ['cat', 'dog'],
                        help='operation mode, cat or dog.')

    parser.add_argument('-seed',
                        default=123,
                        required=False,
                        help='seed for random number generation.')

    parser.add_argument('-data_dir',
                        default='../cat-and-dog/',
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

'''
Funtions that used to initialize the model. There are two working modes: feature_extracting and fine_tuning.
In feature_extracting, only the last fully connected layer is updated. In fine_tuning, all the layers are
updated.
'''
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return model

# Parameters of newly constructed modules have requires_grad=True by default
def make_model(modelName, feature_extracting, use_pretrained):
    if modelName == 'resnet18':
        model = models.resnet18(pretrained = use_pretrained)
        model = set_parameter_requires_grad(model, feature_extracting)
        model.fc = nn.Linear(2048, 1)
        return model
    elif modelName == 'resnet50':
        model = models.resnet50(pretrained = use_pretrained)
        model = set_parameter_requires_grad(model, feature_extracting)
        model.fc = nn.Linear(8192, 1)
        return model
    elif modelName == "vgg11":
        model = models.vgg11(pretrained=use_pretrained)
        model = set_parameter_requires_grad(model, feature_extracting)
        model.classifier[0] = nn.Linear(32768,4096)
        model.classifier[3] = nn.Linear(4096,4096)
        model.classifier[6] = nn.Linear(4096, 1)
        return model
    elif modelName == "vgg11_bn":
        model = models.vgg11_bn(pretrained=use_pretrained)
        model.classifier[0] = nn.Linear(32768,4096)
        model.classifier[3] = nn.Linear(4096,4096)
        model.classifier[6] = nn.Linear(4096, 1)
        return model
    else:
        print("Invalid model name, exiting...")
        exit()
#-----------------------------------------------------------------------------------------------------------------
def cats_vs_dogs_config(device):
    """
    Configurations and hyper-parameters for casts-vs-dogs
    """
    # model name
    modelName = 'resnet18'

    # if true, only train the fully connected layers
    featureExtracting = False
    print('Feature extracting: '+str(featureExtracting))

    # whether to use pretrained model
    usePretrained = True
    print('Using pretrained model: '+str(usePretrained))

    # initialize the model
    net = make_model(modelName, feature_extracting = featureExtracting, use_pretrained = usePretrained)
    print(str(net))
    # if there are multiple GPUs, split the batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)
    # send model to GPU
    net = net.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = net.parameters()
    print("Params to learn:")
    if featureExtracting:
        params_to_update = []
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print(str(name))
    else: # if finetuning
        for name,param in net.named_parameters():
            if param.requires_grad == True:
                print(str(name))

    # loss function.
    # calulating weight for loss function because number of control datapoints is much larger
    # than heme datapoints. See docs for torch.nn.BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005, amsgrad=False)
    #optimizer = optim.SGD(params_to_update, lr=0.0005, weight_decay=0.01, momentum=0)
    print('optimizer:')
    print(str(optimizer))
    
    # whether to decay the learning rate
    learningRateDecay = True

    # number of epochs to train the Neural Network
    num_epochs = 5

    # learning rate schduler used to implement learning rate decay
    learningRateScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=0.1)

    # a dictionary contains optimizer, learning rate scheduler
    # and whether to decay learning rate
    optimizerDict = {'optimizer': optimizer,
                    'learningRateScheduler':learningRateScheduler,
                    'learningRateDecay':learningRateDecay}

    return net, loss_fn, optimizerDict, num_epochs

def dataPreprocess(op, batch_size, data_dir, seed, normalize):
    """
    Function to pre-process the data: load data, normalize data, random split data
    with a fixed seed. Return the size of input data points and DataLoaders for training,
    validation and testing.
    The mean and std are from the dataset_statistics.py from the entire dataset.
    """    
    # seed for random splitting dataset
    torch.manual_seed(seed)

    # dataset statistics
    if op == 'cat':
        mean = [0.4728, 0.4466, 0.4013]
        std = [0.2140, 0.2109, 0.2096]
    elif op== 'dog':
        mean = [0.4757, 0.4523, 0.4007]
        std = [0.2111, 0.2057, 0.2036]       

    # define the transforms we need
    # Load the training data again with normalization if needed.
    image_size = (256, 256)  
    if normalize == True:  
        print( 'normalizing data, mean:'+str(mean)+', std:'+str(std))
        transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((mean[0], mean[1], mean[2]),(std[0], std[1], std[2]))
                                       ])
    else:
        print( 'NOT normalizing data.')
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    # load training set
    trainset = torchvision.datasets.ImageFolder(data_dir+'train/',
                                               transform=transform,
                                               target_transform=None)

    # load testing set
    testset = torchvision.datasets.ImageFolder(data_dir+'val/',
                                               transform=transform,
                                               target_transform=None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=16)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=116)

    return trainloader, testloader

def train_model(model, device, dataloaders, criterion, optimizerDict, num_epochs):
    """
    The train_model function handles the training and validation of a given model.
    As input, it takes 'a PyTorch model', 'a dictionary of dataloaders', 'a loss function', 'an optimizer',
    'a specified number of epochs to train and validate for'. The function trains for the specified number
    of epochs and after each epoch runs a full validation step. It also keeps track of the best performing
    model (in terms of validation accuracy), and at the end of training returns the best performing model.
    After each epoch, the training and validation accuracies are printed.    
    """
    since = time.time()
    sigmoid = nn.Sigmoid()
    optimizer = optimizerDict['optimizer']
    learningRateScheduler = optimizerDict['learningRateScheduler']
    learningRateDecay = optimizerDict['learningRateDecay']
    if learningRateDecay:
        print('Using learning rate scheduler.')
    else:
        print('Not using andy learning rate scheduler.')

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    train_mcc_history = []
    val_mcc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mcc = 0.0

    for epoch in range(num_epochs):
        print(' ')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 15)

        epoch_labels_train = []
        epoch_labels_val = []
        epoch_labels = {'train':epoch_labels_train, 'val':epoch_labels_val}
        epoch_preds_train = []
        epoch_preds_val = []
        epoch_preds = {'train':epoch_preds_train, 'val':epoch_preds_val}
        epoch_outproba_val = [] # output probabilities in validation phase

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # accumulated loss for a epoch
            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # Get model outputs and calculate loss
                #labels = Variable(labels)
                labels = labels.view(-1,1)

                # calculate output of neural network
                outputs = model(inputs)

                # output probabilities
                outproba = sigmoid(outputs)

                loss = criterion(outputs, labels.float())

                # sigmoid output,  preds>0.0 means sigmoid(preds)>0.5
                preds = (outputs > 0.0)
                #print( 'outpus:',outputs)
                #print( 'preds:',preds)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward() # back propagate
                    optimizer.step() # update parameters
                    optimizer.zero_grad() # zero the parameter gradients

                # accumulate loss for this single batch
                running_loss += loss.item() * inputs.size(0)

                # concatenate the list
                # need to conver from torch tensor to numpy array(n*1 dim),
                # then squeeze into one dimensional vectors
                labels = labels.cpu() # cast to integer then copy tensors to host
                preds = preds.cpu() # copy tensors to host
                outproba = outproba.cpu()
                epoch_labels[phase] = np.concatenate((epoch_labels[phase],np.squeeze(labels.numpy())))
                epoch_preds[phase] = np.concatenate((epoch_preds[phase],np.squeeze(preds.numpy())))

                # only collect validation phase's output probabilities
                if phase == 'val':
                    epoch_outproba_val = np.concatenate((epoch_outproba_val,np.squeeze(outproba.detach().numpy())))

            # loss for the epoch, averaged over all the data points
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            #print( epoch_labels[phase].astype(int))
            #print( epoch_preds[phase].astype(int))
            #print( epoch_labels[phase].astype(int).shape)
            #print( epoch_preds[phase].astype(int).shape)

            # calculate metrics using scikit-learn
            epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc = calc_metrics(epoch_labels[phase].astype(int),epoch_preds[phase].astype(int))
            print('{} Accuracy:{:.3f} Precision:{:.3f} Recall:{:.3f} F1:{:.3f} MCC:{:.3f}.'.format(phase, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_mcc))

            # record the training accuracy
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                train_mcc_history.append(epoch_mcc)

            # record the validation accuracy
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_mcc_history.append(epoch_mcc)

            # deep copy the model with best mcc
            if phase == 'val' and epoch_mcc > best_mcc:
                best_mcc = epoch_mcc
                best_model_wts = copy.deepcopy(model.state_dict())
                # ROC curve when mcc is best
                fpr, tpr, thresholds = metrics.roc_curve(epoch_labels[phase], epoch_outproba_val)
                # print to list so that can be saved in .json file
                fpr = fpr.tolist()
                tpr = tpr.tolist()
                thresholds = thresholds.tolist()
                best_mcc_roc = {'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}

        # decay the learning rate at end of each epoch
        if learningRateDecay:
            learningRateScheduler.step()

    time_elapsed = time.time() - since
    print( 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print( 'Best val MCC: {:4f}'.format(best_mcc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_mcc_roc, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history
