
'''
Cross validate cats vs control or dogs vs control
'''

import torch
import json
from utils import getArgs
from utils import dataPreprocess
from utils import train_model
from utils import cats_vs_dogs_config

# get input parameters for the program
args = getArgs()
op = args.op
seed = args.seed
data_dir = args.data_dir
batch_size = args.batch_size
print('batch size: '+str(batch_size))
normalizeDataset = args.normalizeDataset


# calculate positive weight for loss function
if op == 'cat':
    print('performing cats vs control cross-validation task.')
elif op == 'dog':
    print('performing dogs vs control cross-validation task.')
print('data location:'+str(data_dir))

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current device: '+str(device))

# lists of the results of 10 folds, each element of the list is a list containing the
# corresponding result of 1 fold
train_acc_history_list=[]
val_acc_history_list=[]
train_loss_history_list=[]
val_loss_history_list=[]
train_mcc_history_list=[]
val_mcc_history_list=[]
best_mcc_roc_list=[]

# perfrom 10-fold cross-validation
for i in range(5):
    print('*********************************************************************')
    print('starting {}th fold cross-validation'.format(i+1))

    # import the model and training configurations
    net, loss_fn, optimizerDict, num_epochs = cats_vs_dogs_config(device)

    # put data into 2 DataLoaders
    data_dir_cv = data_dir + 'cv' + str(i+1) + '/'
    trainloader, valloader = dataPreprocess(op, batch_size, data_dir_cv, seed, normalizeDataset)

    # create the data loader dictionary
    dataloaders_dict = {"train": trainloader, "val": valloader}

    # train the neural network
    trained_model, best_mcc_roc, train_acc_history, val_acc_history, train_loss_history, val_loss_history, train_mcc_history, val_mcc_history = train_model(net,
                                                    device,
                                                    dataloaders_dict,
                                                    loss_fn,
                                                    optimizerDict,
                                                    num_epochs = num_epochs
                                                    )
    # pust the histories into the lists
    train_acc_history_list.append(train_acc_history)
    val_acc_history_list.append(val_acc_history)
    train_loss_history_list.append(train_loss_history)
    val_loss_history_list.append(val_loss_history)
    train_mcc_history_list.append(train_mcc_history)
    val_mcc_history_list.append(val_mcc_history)
    best_mcc_roc_list.append(best_mcc_roc)

lists_to_save = [train_acc_history_list,
                 val_acc_history_list,
                 train_loss_history_list,
                 val_loss_history_list,
                 train_mcc_history_list,
                 val_mcc_history_list,
                 best_mcc_roc_list]

resultFile = './log/'+op+'_cv'+'.json'
with open(resultFile, 'w') as fp:
    json.dump(lists_to_save, fp)

print('cross-validation finished, end of program')