#!/usr/bin/env python
# coding: utf-8

"""
This code is modified from the code found in https://arxiv.org/abs/2011.07727. This generates an masked shallow auto-encoder from the training snapshot, "./data/snapshot_git.p" with parameters values as printed. 
The auto-encoder is save at './model/AE_git.tar'. This is used in the LaSDI_1DBurgers_NM.ipynb notebook.

Last Modified: Bill Fries 2/2/22

"""


import torch
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils

import numpy as np
import numpy.linalg as LA

import sys,time,os
import pickle

sys.path.append("..")
import modAutoEncoder as autoencoder
import modLaSDIUtils as utils


# In[2]:


get_ipython().system('nvidia-smi')


# ## Training the Auto-Encoder
# We now train the auto-encoder as per the "train_NM-ROM_swish" file. We use the snapshot data from above to train the neural network. 


# Choose device that is not being used
gpu_ids = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

# In[4]:
nx = 1001
dx = 6 / (nx - 1)
nt = 1000
tstop = 1
x=np.linspace(-3, 3, nx)

dt = tstop / nt 
c = dt/dx
t = np.linspace(0, tstop, nt)

# load solution snapshot
solution_snapshot_orig = pickle.load(open("./data/snapshot_git.p", 'rb'))

# substract IC -> centered on IC
ndata=solution_snapshot_orig.shape[0]
nset= round(ndata/(nt+1))

solution_snapshot=np.array([])
for foo in range(nset):
    solution_snapshot=np.append(solution_snapshot,solution_snapshot_orig[foo*(nt+1):(foo+1)*(nt+1)])#    -solution_snapshot_orig[foo*(nt+1)])

solution_snapshot=np.reshape(solution_snapshot,(-1,nx))

# remove BC
solution_snapshot = solution_snapshot[:,:-1].astype('float32')

# define testset and trainset indices
test_ind = np.random.permutation(np.arange(solution_snapshot.shape[0]))[:int(0.1*solution_snapshot.shape[0])]
train_ind = np.setdiff1d(np.arange(solution_snapshot.shape[0]),test_ind)

# set trainset and testset
trainset = solution_snapshot[train_ind]
testset = solution_snapshot[test_ind] 

# set dataset
dataset = {'train':data_utils.TensorDataset(torch.tensor(trainset)),
           'test':data_utils.TensorDataset(torch.tensor(testset))}
print(dataset['train'].tensors[0].shape, dataset['test'].tensors[0].shape)

# compute dataset shapes
dataset_shapes = {'train':trainset.shape,
                 'test':testset.shape}

print(dataset_shapes['train'],dataset_shapes['test'])

# set device
device = autoencoder.getDevice()
print("Using device:", device, '\n')

# set encoder and decoder types, activation function, etc.
encoder_class = autoencoder.Encoder
decoder_class = autoencoder.Decoder
f_activation = autoencoder.SiLU

# set the number of nodes in each layer
m = 1000
f = 4
b = 36
db = 12
M2 = b + (m-1)*db
M1 = 2*m
mask = utils.create_mask_1d(m,b,db)

# set batch_size, number of epochs, paitience for early stop
batch_size = 20
num_epochs = 1000
num_epochs_print = num_epochs//100
early_stop_patience = num_epochs//10

# autoencoder filename
AE_fname = 'model/AE_git.tar'
chkpt_fname = 'checkpoint.tar'

encoder, decoder = autoencoder.createAE(encoder_class,
                                        decoder_class,
                                        f_activation,
                                        mask,
                                        m, f, M1, M2,
                                        device )

# set data loaders
train_loader = DataLoader(dataset=dataset['train'], 
                          batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=dataset['test'], 
                         batch_size=batch_size, shuffle=True, num_workers=0)
data_loaders = {'train':train_loader, 'test':test_loader}

# train
autoencoder.trainAE(encoder,
                    decoder,
                    trainset,
                    testset,
                    batch_size,
                    num_epochs,
                    num_epochs_print,
                    early_stop_patience,
                    AE_fname,
                    chkpt_fname )
