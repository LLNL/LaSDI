#!/usr/bin/env python
# coding: utf-8

"""
This code is modified from the code found in https://arxiv.org/abs/2011.07727. This generates an masked shallow auto-encoder from the training snapshot, "./data/snapshot_git.p" with parameters values as printed. 
The auto-encoder is save at './model/AE_git.tar'. This is used in the LaSDI_1DBurgers_NM.ipynb notebook.

Last Modified: Bill Fries 2/2/22

"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
from torch.optim import lr_scheduler

import numpy as np
import numpy.linalg as LA

from scipy import sparse as sp
from scipy import sparse
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
from scipy.io import savemat,loadmat
import scipy.integrate as integrate

from itertools import combinations_with_replacement, product
from sklearn.decomposition import SparseCoder
import sys,time
import os
import copy
import pickle


# In[2]:


get_ipython().system('nvidia-smi')


# ## Training the Auto-Encoder
# We now train the auto-encoder as per the "train_NM-ROM_swish" file. We use the snapshot data from above to train the neural network. 

# In[3]:


# Choose device that is not being used
gpu_ids = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

def create_mask_v2(m,b,db):
    '''
    mask=create_mask_v2(m,b,db)  
    '''
    
    M2 = b + db*(m-1)
    mask = np.zeros((m,M2),dtype='int')
    
    block = np.ones(b,dtype='int')
    ind = np.arange(b)
    for row in range(m):
        col = ind + row*db
        mask[row,col] = block
    
    print(
        "Sparsity in {} by {} mask: {:.2f}%".format(
            m, M2, (1.0-np.count_nonzero(mask)/mask.size)*100
        )
    )
    
    plt.figure()
    plt.spy(mask)
    plt.show()

    return mask


def silu(input):
    return input * torch.sigmoid(input)
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return silu(input)
    
class Encoder(nn.Module):
    def __init__(self,m,M1,f):
        super(Encoder,self).__init__()
        self.full = nn.Sequential(
            nn.Linear(m,M1),
            SiLU(),
            nn.Linear(M1,f,bias=False)
        )
                
    def forward(self, y):     
        y = y.view(-1,m)
        T = self.full(y)
        T = T.squeeze()
        
        return T
    
class Decoder(nn.Module):
    def __init__(self,f,M2,m):
        super(Decoder,self).__init__()
        self.full = nn.Sequential(
            nn.Linear(f,M2,bias=False),
            SiLU(),
            nn.Linear(M2,m,bias=False)
        )
             
    def forward(self,T):
        T = T.view(-1,f)
        y = self.full(T)
        y = y.squeeze()
        
        return y


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


# set batch_size, number of epochs, paitience for early stop
batch_size = 20
num_epochs = 1000
num_epochs_print = num_epochs//100
early_stop_patience = num_epochs//10


# set data loaders
train_loader = DataLoader(dataset=dataset['train'], 
                          batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=dataset['test'], 
                         batch_size=batch_size, shuffle=True, num_workers=0)

data_loaders = {'train':train_loader, 'test':test_loader}


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, '\n')


# In[5]:



# set checkpoint
PATH = './checkpoint_v2.tar'


# set the number of nodes in each layer
m = 1000
f = 4
b = 36
db = 12
M2 = b + (m-1)*db
M1 = 2*m


# load model
try:
    checkpoint = torch.load(PATH, map_location=device)
    
    encoder = Encoder(m,M1,f).to(device)
    decoder = Decoder(f,M2,m).to(device)
    
    # Prune
    mask = create_mask_v2(m,b,db)
    prune.custom_from_mask(decoder.full[2], name='weight', mask=torch.tensor(mask).to(device))    
    
#     optimizer = torch.optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()), lr=1)
#     scheduler = None     
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=10) 
    
    loss_func = nn.MSELoss(reduction='mean')
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    last_epoch = checkpoint['epoch']
    loss_hist = checkpoint['loss_hist']
    best_loss = checkpoint['best_loss']
    early_stop_counter = checkpoint['early_stop_counter']
    best_encoder_wts = checkpoint['best_encoder_wts']
    best_decoder_wts = checkpoint['best_decoder_wts']
    
    print("\n--------checkpoint restored--------\n")
    
    # compute sparsity in mask
    mask = decoder.state_dict()['full.2.weight_mask']
    print("Sparsity in {} by {} mask: {:.2f}%".format(
        mask.shape[0], mask.shape[1], 100. * float(torch.sum(mask == 0))/ float(mask.nelement())))

    # resume training
    print("")
    print('Re-start {}th training... m={}, f={}, M1={}, M2={}, b={}, db={}'.format(
        last_epoch+1, m, f, M1, M2, b, db))
except:
    encoder = Encoder(m,M1,f).to(device)
    decoder = Decoder(f,M2,m).to(device)
    
    # Prune
    mask = create_mask_v2(m,b,db)
    prune.custom_from_mask(decoder.full[2], name='weight', mask=torch.tensor(mask).to(device))

#     optimizer = torch.optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()), lr=1)
#     scheduler = None 
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=10) 
    
    loss_func = nn.MSELoss(reduction='mean')
    
    last_epoch = 0
    loss_hist = {'train':[],'test':[]}
    best_loss = float("inf")
    early_stop_counter = 1
    best_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_decoder_wts = copy.deepcopy(decoder.state_dict())
    
    print("\n--------checkpoint not restored--------\n")
    
    # compute sparsity in mask
    mask = decoder.state_dict()['full.2.weight_mask']
    print("Sparsity in {} by {} mask: {:.2f}%".format(
        mask.shape[0], mask.shape[1], 100. * float(torch.sum(mask == 0))/ float(mask.nelement())))

    # start training
    print("")
    print('Start first training... m={}, f={}, M1={}, M2={}, b={}, db={}'.format(
        m, f, M1, M2, b, db))
pass


# train model
since = time.time()

for epoch in range(last_epoch+1,num_epochs+1):   

    if epoch%num_epochs_print == 0:
        print()
        if scheduler !=None:
            print('Epoch {}/{}, Learning rate {}'.format(
                epoch, num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            print('Epoch {}/{}'.format(
                epoch, num_epochs))
        print('-' * 10)

    # Each epoch has a training and test phase
    for phase in ['train', 'test']:
        if phase == 'train':
            encoder.train()  # Set model to training mode
            decoder.train()  # Set model to training mode
        else:
            encoder.eval()   # Set model to evaluation mode
            decoder.eval()   # Set model to evaluation mode
            
        running_loss = 0.0

        # Iterate over data
        for data, in data_loaders[phase]:
            inputs = data.to(device)
            targets = data.to(device)

            if phase == 'train':
                if scheduler != None:
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = decoder(encoder(inputs))
                    loss = loss_func(outputs, targets)

                    # backward
                    loss.backward()

                    # optimize
                    optimizer.step()  
                    
                    # add running loss
                    running_loss += loss.item()*inputs.shape[0]
                else:
                    def closure():
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        outputs = decoder(encoder(inputs))
                        loss = loss_func(outputs,targets)

                        # backward
                        loss.backward()
                        return loss

                    # optimize
                    optimizer.step(closure)
                    
                    # add running loss
                    with torch.set_grad_enabled(False):
                        outputs = decoder(encoder(inputs))
                        running_loss += loss_func(outputs,targets).item()*inputs.shape[0]
                    
            else:
                with torch.set_grad_enabled(False):
                    outputs = decoder(encoder(inputs))
                    running_loss += loss_func(outputs,targets).item()*inputs.shape[0]

        # compute epoch loss
        epoch_loss = running_loss / dataset_shapes[phase][0]
        loss_hist[phase].append(epoch_loss)
            
        # update learning rate
        if phase == 'train' and scheduler != None:
            scheduler.step(epoch_loss)

        if epoch%num_epochs_print == 0:
            print('{} MSELoss: {}'.format(
                phase, epoch_loss))

    # deep copy the model
    if loss_hist['test'][-1] < best_loss:
        best_loss = loss_hist['test'][-1]
        early_stop_counter = 1
        best_encoder_wts = copy.deepcopy(encoder.state_dict())
        best_decoder_wts = copy.deepcopy(decoder.state_dict())
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:  
            break
    
    # save checkpoint every num_epoch_print
    if epoch%num_epochs_print== 0:
        torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_hist': loss_hist,
                    'best_loss': best_loss,
                    'early_stop_counter': early_stop_counter,
                    'best_encoder_wts': best_encoder_wts,
                    'best_decoder_wts': best_decoder_wts,
                    }, PATH)        

print()
print('Epoch {}/{}, Learning rate {}'      .format(epoch, num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
print('-' * 10)
print('train MSELoss: {}'.format(loss_hist['train'][-1]))
print('test MSELoss: {}'.format(loss_hist['test'][-1]))

time_elapsed = time.time() - since

# load best model weights
encoder.load_state_dict(best_encoder_wts)
decoder.load_state_dict(best_decoder_wts)

# compute best train MSELoss
# encoder.to('cpu').eval()
# decoder.to('cpu').eval()

with torch.set_grad_enabled(False):
    train_inputs = torch.tensor(trainset).to(device)
    train_targets = torch.tensor(trainset).to(device)
    train_outputs = decoder(encoder(train_inputs))
    train_loss = loss_func(train_outputs,train_targets).item()

# print out training time and best results
print()
if epoch < num_epochs:
    print('Early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'          .format(epoch-last_epoch, time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
else:
    print('No early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'          .format(epoch-last_epoch, time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
print('-' * 10)
print('Best train MSELoss: {}'.format(train_loss))
print('Best test MSELoss: {}'.format(best_loss))

###### save models ########
print()
print("Saving after {}th training to".format(epoch),
      './model/AE_git.tar')
torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, './model/AE_git.tar')


# plot train and test loss
plt.figure()
plt.semilogy(loss_hist['train'])
plt.semilogy(loss_hist['test'])
plt.legend(['train','test'])
plt.show()   

# delete checkpoint
try:
    os.remove(PATH)
    print()
    print("checkpoint removed")
except:
    print("no checkpoint exists")
    
torch.cuda.empty_cache()





