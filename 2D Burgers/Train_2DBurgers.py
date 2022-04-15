#!/usr/bin/env python
# coding: utf-8

"""
This code is modified from the code found in https://arxiv.org/abs/2011.07727. This generates an masked shallow auto-encoder from the training snapshot, "./data/snapshot_git.p" with parameters values as printed. 
The auto-encoders are saved at './model/AE_u_git.tar' and './model/AE_v_git.tar' respectively. This is memory intensive as there are many parameters in the NNs. This is used in the LaSDI_2DBurgers_NM.ipynb notebook.

Last Modified: Bill Fries 2/2/22

"""
import sys, os
import pickle

import numpy as np
import torch.nn as nn

sys.path.append("..")
import modAutoEncoder as autoencoder
import modLaSDIUtils as utils

# Set print option
np.set_printoptions(threshold=sys.maxsize)

# Choose device that is not being used
gpu_ids = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

# Set device
device = autoencoder.getDevice()
print("Using device:", device, '\n')

# Given parameters
nx = 60
ny = 60
m = (ny-2)*(nx-2) # 3364
nt = 1500

# Choose either Re=10000 or Re=100
Re = 10000 
    
# Choose data normalize option (option 1: -1<=X<=1 option 2: 0<=X<=1)
option = 2

# Choose activation function (sigmoid, swish)
activation = 'sigmoid'

# set batch_size, number of epochs, paitience for early stop
batch_size = 1000
num_epochs = 30000//5
num_epochs_print = num_epochs//100
early_stop_patience = num_epochs//50

encoder_class = autoencoder.Encoder
decoder_class = autoencoder.Decoder
if activation=='sigmoid':
  f_activation = nn.Sigmoid
elif activation=='swish':
  f_activation = autoencoder.SiLU
else:
    raise NameError('{} is given for option, but it must be either sigmoid or swish'.format(activation))


LS_dim = 3

print('')
print("Beginning Latent Space Dimension: {}".format(LS_dim))
print('')
# load snapshot
if Re==10000:
    file_name_snapshot="./data/snapshot_git.p"
elif Re==100:
    file_name_snapshot="./data/snapshot_full_low_Re.p"
else:
    raise NameError('{} is given for Re, but it must be either 100 or 10000'.format(Re)) 

snapshot = pickle.load(open(file_name_snapshot,'rb'))

# number of data points
snapshot_u = snapshot['u'].reshape(-1,nx*ny).astype('float32')
snapshot_v = snapshot['v'].reshape(-1,nx*ny).astype('float32')
ndata = snapshot_u.shape[0]
# remove BC
multi_index_i,multi_index_j=np.meshgrid(np.arange(nx),np.arange(ny),indexing='xy')
full_multi_index=(multi_index_j.flatten(),multi_index_i.flatten())
free_multi_index=(multi_index_j[1:-1,1:-1].flatten(),multi_index_i[1:-1,1:-1].flatten())

dims=(ny,nx)
full_raveled_indicies=np.ravel_multi_index(full_multi_index,dims)
free_raveled_indicies=np.ravel_multi_index(free_multi_index,dims)

orig_data_u = snapshot_u[:,free_raveled_indicies]
orig_data_v = snapshot_v[:,free_raveled_indicies]

data_u = orig_data_u
data_v = orig_data_v

# check shapes of snapshot
print('data shape')
print(data_u.shape)
print(data_v.shape)

# generate mesh grid
[xv,yv]=np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,ny),indexing='xy')
x=xv.flatten()
y=yv.flatten()

x_free=x[free_raveled_indicies]
y_free=y[free_raveled_indicies]

k=0

# define testset and trainset indices
nset = round(ndata/(nt+1))
test_ind = np.array([],dtype='int')
for foo in range(nset):
    rand_ind = np.random.permutation(np.arange(foo*(nt+1)+1,(foo+1)*(nt+1)))[:int(0.1*(nt+1))]
    test_ind = np.append(test_ind,rand_ind)
train_ind = np.setdiff1d(np.arange(ndata),test_ind)

# set trainset and testset
trainset_u = data_u[train_ind]
trainset_v = data_v[train_ind]
testset_u = data_u[test_ind] 
testset_v = data_v[test_ind] 

# print dataset shapes
print('trainset_u shape: ', trainset_u.shape)
print('trainset_v shape: ', trainset_v.shape)
print('testset_u shape: ', testset_u.shape)
print('testset_v shape: ', testset_v.shape)

# set the number of nodes in each layer
a = 2
b = int(100)
db = int(10)

M1 = int(a*m) # encoder hidden layer
M2 = b + (m-1)*db # decoder hidden layer

f = LS_dim # latent dimension

# sparsity and shape of mask
mask_2d=utils.create_mask_2d((nx-2),(ny-2),m,b,db)

# number of parameters and memory
en_para=m*M1+M1+M1*f
de_para=f*M2+M2+np.count_nonzero(mask_2d)
print('Encoder parameters:{:.8e}({:.4}GB)'.format(en_para,en_para*4/2**30),      'Decoder parameters:{:.8e}({:.4}GB)'.format(de_para,(f*M2+M2+M2*m)*4/2**30))

# data size
data_size=np.prod(orig_data_u.shape)
print('Data size:{:.8e}({:.4}GB)'.format(data_size,data_size*4/2**30))

# Set file names
if Re==10000:
    file_name_AE_u= './model/AE_u_git.tar'
    file_name_AE_v= './model/AE_v_git.tar'
elif Re==100:
    file_name_AE_u="./model/AE_u_low_Re_v3_batch_240.pkl"
    file_name_AE_v="./model/AE_v_low_Re_v3_batch_240.pkl"
else:
    raise NameError('{} is given for Re, but it must be either 100 or 10000'.format(Re))  
file_name_chkpt_u = 'checkpoint_u.tar'
file_name_chkpt_v = 'checkpoint_v.tar'

# ## For u
encoder_u, decoder_u = autoencoder.createAE(encoder_class,
                                            decoder_class,
                                            f_activation,
                                            mask_2d,
                                            m, f, M1, M2,
                                            device )
# train
autoencoder.trainAE(encoder_u,
                    decoder_u,
                    trainset_u,
                    testset_u,
                    batch_size,
                    num_epochs,
                    num_epochs_print,
                    early_stop_patience,
                    file_name_AE_u,
                    file_name_chkpt_u )


# ## For v
encoder_v, decoder_v = autoencoder.createAE(encoder_class,
                                            decoder_class,
                                            f_activation,
                                            mask_2d,
                                            m, f, M1, M2,
                                            device )
# train
autoencoder.trainAE(encoder_v,
                    decoder_v,
                    trainset_v,
                    testset_v,
                    batch_size,
                    num_epochs,
                    num_epochs_print,
                    early_stop_patience,
                    file_name_AE_v,
                    file_name_chkpt_v )
