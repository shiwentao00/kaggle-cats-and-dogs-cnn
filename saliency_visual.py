
'''
    Generate a saliency map for an image to find the "implicit attention" that
    the neural networks are paying to. This method is introduced by the paper
    "Deep Inside Convolutional Networks: Visualising Image Classification
    Models and Saliency Maps".
    We use this module to see which part of the protein is important to the neural
    network to distinguish them.
'''

# import libraries
import torch
import torchvision
import os
from torchvision import datasets
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from utils import make_model
from helper import imshow
def getArgs():
    """
    The parser function to set the default vaules of the program parameters or read the new value set by user.
    """    
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-opMode',
                        default='cats_vs_dogs',
                        required=False,
                        choices = ['cats_vs_dogs', 'cats_vs_control', 'dogs_vs_contorl'],
                        help= 'choices: cats_vs_dogs, cats_vs_control, dogs_vs_contorl')
    
    parser.add_argument('-index',
                        type=int,
                        default=1,
                        required=False,
                        help='index of the target image')

    parser.add_argument('-data_dir_cats_vs_dogs',
                        default='../cat-and-dog/',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_cats_vs_control',
                        default='../cat-and-control',
                        required=False,
                        help='directory to load data for training, validating and testing.')

    parser.add_argument('-data_dir_dogs_vs_control',
                        default='../dog-and-control/',
                        required=False,
                        help='directory to load data for 10-fold cross-validation.')

    parser.add_argument('-model_cats_vs_dogs',
                        default='./model/cats_vs_dogs_resnet18.pt',
                        required=False,
                        help='file to save the model for cats vs dogs.')

    parser.add_argument('-model_cats_vs_control',
                        default='./model/cats_vs_control_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_nucleotide.')

    parser.add_argument('-model_dogs_vs_control',
                        default='./model/dogs_vs_control_resnet18.pt',
                        required=False,
                        help='file to save the model for control_vs_nucleotide.')
    return parser.parse_args()

# define our own dataset class.
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# the function returns the saliency map of an image
def saliencyMap(model, image, device, classes):
    # set model to evaluation mode
    model.eval()

    # send image to gpu
    image.to(device)

    # don't calculate gradients for the parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # need to calculate gradients of the input image
    image.requires_grad = True

    # output of the neural network, as in the paper, don't pass the output
    # to softmax or sigmoid
    outputs = model(image)

    # get the gradients of the input
    outputs.backward()
    saliencyMap = image.grad
    #print('size of gradients:',saliencyMap.shape)

    # convert back to numpy
    saliencyMap = saliencyMap.cpu()
    saliencyMap = np.squeeze(saliencyMap.numpy())
    #print('size of gradients:',saliencyMap.shape)
    # absolute values of the pixels
    saliencyMap = np.absolute(saliencyMap)
    #print('size of gradients:',saliencyMap.shape)
    # max value among the channels
    # dim 0: channel, dim 1, 2: height & width
    saliencyMap = np.amax(saliencyMap, axis=0)
    #print('size of gradients:',saliencyMap.shape)

    # the prediction
    predIdx = outputs.detach().cpu().numpy().squeeze().item()
    predIdx = (predIdx > 0)
    predClass = classes[int(predIdx)]
    return saliencyMap, predClass

# normalize the saliency map to have more contrast
def saliencyNorm(saliencyMap):
    maxPixel = np.amax(saliencyMap)
    minPixel = np.amin(saliencyMap)
    rangePixel = maxPixel - minPixel
    #print('range of pixels:',rangePixel)
    return np.true_divide(saliencyMap, rangePixel)

# returns original image, saliency map and file path
def imgGen(dataset, datasetWithNorm, index, classes, model, device):
    # get an image to display (256*256*3)
    imageDspl, labelDspl, pathDspl  = dataset.__getitem__(index)
    imageDspl = imageDspl.unsqueeze(0)
    imageDspl = imageDspl.squeeze().detach().numpy()
    imageDspl = np.transpose(imageDspl, (1, 2, 0))

    # get an normalized image to send to model
    image, label, path = datasetWithNorm.__getitem__(index)
    image = image.unsqueeze(0)

    # double check
    assert(path == pathDspl)
    assert(labelDspl == label)
    imgClass = classes[label]
    fileName = os.path.basename(path)

    # generate saliency map (256*256)
    imageSaliency, predClass = saliencyMap(model, image, device, classes)
    imageSaliency = saliencyNorm(imageSaliency)
    return imageDspl, imageSaliency, fileName, imgClass, predClass

if __name__ == "__main__":
    args = getArgs()
    opMode = args.opMode
    if opMode == 'cats_vs_dogs':
        data_dir = args.data_dir_cats_vs_dogs
        model_dir = args.model_cats_vs_dogs
        classes = ('cat', 'dog')
    elif opMode == 'cats_vs_control':
        data_dir = args.data_dir_cats_vs_control
        model_dir = args.model_cats_vs_control
        classes = ('control', 'cat')
    elif opMode == 'dogs_vs_control':
        data_dir = args.data_dir_dogs_vs_control
        model_dir = args.model_dogs_vs_control
        ('control', 'dog')
    index = args.index
    batch_size= 1
    print('batch size: '+str(batch_size))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    # dataset statistics
    mean_cats_vs_dogs = [0.4883, 0.4551, 0.4174]
    std_cats_vs_dogs = [0.2265, 0.2214, 0.2220]
    if opMode == 'cats_vs_dogs':
        mean = mean_cats_vs_dogs
        std = std_cats_vs_dogs

    # initialize model
    modelName = 'resnet18'
    feature_extracting = False
    net = make_model(modelName, feature_extracting = feature_extracting, use_pretrained = True)
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        net = nn.DataParallel(net)

    # send model to GPU
    net = net.to(device)

    # Load trained parameters
    net.load_state_dict(torch.load(model_dir))

    # define a transform with mean and std, another transform without them.
    image_size = (256, 256)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    transformWithNorm = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((mean[0], mean[1], mean[2]),(std[0], std[1], std[2]))
                                       ])

    dataset = ImageFolderWithPaths(data_dir+'test_set',transform=transform,target_transform=None)
    datasetWithNorm = ImageFolderWithPaths(data_dir+'test_set',transform=transformWithNorm,target_transform=None)

    imageDspl, imageSaliency, fileName, imgClass, predClass = imgGen(dataset, datasetWithNorm, index, classes, net, device)

    fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14, 7))
    ax1.imshow(imageDspl)
    ax1.set_title('Original picture')
    ax2.imshow(imageSaliency)
    ax2.set_title('Saliency maps')
    fig.suptitle('task:'+opMode+',type:'+imgClass+',filename: '+ fileName+' predicted class:'+predClass)
    # show both figures
    plt.savefig('./images/'+str(opMode)+'_'+str(imgClass)+str(index)+'.png')
    plt.show()