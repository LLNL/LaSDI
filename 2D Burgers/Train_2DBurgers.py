#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
from torch.optim import lr_scheduler
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys,time
import os
import copy
import pickle


# In[15]:


# Set print option
np.set_printoptions(threshold=sys.maxsize)

# Choose device that is not being used
gpu_ids = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, '\n')


# In[16]:


# Given parameters
nx = 60
ny = 60
m = (ny-2)*(nx-2) # 3364
nt = 1500


# In[17]:


# Choose either Re=10000 or Re=100
Re = 10000 
    
# Choose data normalize option (option 1: -1<=X<=1 option 2: 0<=X<=1)
option = 2

# Choose activation function (sigmoid, swish)
activation = 'sigmoid'


# In[18]:


# set batch_size, number of epochs, paitience for early stop
batch_size = 100
num_epochs = 10000//5
num_epochs_print = num_epochs//100
early_stop_patience = num_epochs//50


# In[19]:


# def create_mask(m,b,db):
    
#     M2 = b + db*(m-1)
#     mask = np.zeros((m,M2),dtype='int8')
    
#     block = np.ones(b,dtype='int8')
#     ind = np.arange(b)
#     for row in range(m):
#         col = ind + row*db
#         mask[row,col] = block      

# #     for row in range(nx-2,m):
# #         col = ind + (row-1)*db
# #         mask[row-(nx-2),col] = block    
# #     for row in range(0,m-(nx-2)):
# #         col = ind + (row+1)*db
# #         mask[row+(nx-2),col] = block
                   
#     print(
#         "Sparsity in {} by {} mask: {:.2f}%".format(
#             m, M2, (1.0-np.count_nonzero(mask)/mask.size)*100
#         )
#     )
    
#     plt.figure()
#     plt.spy(mask)
#     plt.show()

#     return mask


# In[20]:


def create_mask_2d(m,b,db):
    
    # local
    Mb=sp.diags([np.ones(nx-2),np.ones(nx-2),np.ones(nx-2)],[0,-1,1],(nx-2,nx-2))
    M=sp.kron(sp.eye(ny-2),Mb,format="csr")

    Ib=sp.eye(nx-2)
    N=sp.kron(sp.diags([np.ones(ny-2),np.ones(ny-2),np.ones(ny-2)],[0,-1,1],(ny-2,ny-2)),Ib,format="csr")

    local=(M+N).astype('int8')
    I,J,V=sp.find(local)
    local[I,J]=1
    
#     col_ind=np.array([],dtype='int')
#     row_ind=np.array([],dtype='int')

#     for lin_ind in range(m):
#         j,i=np.unravel_index(lin_ind,(ny-2,nx-2))

#         E=np.ravel_multi_index((j,np.max((i-1,0))),(ny-2,nx-2))
#         W=np.ravel_multi_index((j,np.min((i+1,nx-2-1))),(ny-2,nx-2))
#         S=np.ravel_multi_index((np.max((j-1,0)),i),(ny-2,nx-2))
#         N=np.ravel_multi_index((np.min((j+1,ny-2-1)),i),(ny-2,nx-2))

#         col=np.unique([lin_ind,E,W,S,N])
#         row=lin_ind*np.ones(col.size,dtype='int')

#         col_ind=np.append(col_ind,col)
#         row_ind=np.append(row_ind,row)

#     data=np.ones(row_ind.size,dtype='int')
#     local2=sp.csr_matrix((data,(row_ind,col_ind)),shape=(m,m))

    # basis
    M2 = int(b + db*(m-1))
    basis = np.zeros((m,M2),dtype='int8')

    block = np.ones(b,dtype='int8')
    ind = np.arange(b)
    for row in range(m):
        col = ind + row*db
        basis[row,col] = block
    
    # mask
    col_ind=np.array([],dtype='int8')
    row_ind=np.array([],dtype='int8')
    for i in range(m):
        col=basis[sp.find(local[i])[1]].sum(axis=0).nonzero()[0]
        row=i*np.ones(col.size)

        col_ind=np.append(col_ind,col)
        row_ind=np.append(row_ind,row)

    data=np.ones(row_ind.size,dtype='int8')
    mask=sp.csr_matrix((data,(row_ind,col_ind)),shape=(m,M2)).toarray()
    
    print(
        "Sparsity in {} by {} mask: {:.2f}%".format(
            m, M2, (1.0-np.count_nonzero(mask)/np.prod(mask.shape))*100
        )
    )
    
    return mask


# In[21]:


if activation=='sigmoid':
    class Encoder(nn.Module):
        def __init__(self,m,M1,f):
            super(Encoder,self).__init__()
            self.full = nn.Sequential(
                nn.Linear(m,M1),
                nn.Sigmoid(),
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
                nn.Linear(f,M2),
                nn.Sigmoid(),
                nn.Linear(M2,m,bias=False)
            )

        def forward(self,T):
            T = T.view(-1,f)
            y = self.full(T)
            y = y.squeeze()

            return y
        
elif activation=='swish':
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
                nn.Linear(f,M2),
                SiLU(),
                nn.Linear(M2,m,bias=False)
            )

        def forward(self,T):
            T = T.view(-1,f)
            y = self.full(T)
            y = y.squeeze()

            return y
else:
    raise NameError('{} is given for option, but it must be either sigmoid or swish'.format(activation))


# In[22]:

for LS_dim in [3]:
    print('')
    print("Beginning Latent Space Dimension: {}".format(LS_dim))
    print('')
    # load snapshot
    if Re==10000:
        file_name_snapshot="../../../../../usr/WS1/fries4/Experiments/2DBurgers/data/snapshot.p"
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


    # In[24]:


    # generate mesh grid
    [xv,yv]=np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,ny),indexing='xy')
    x=xv.flatten()
    y=yv.flatten()

    x_free=x[free_raveled_indicies]
    y_free=y[free_raveled_indicies]

    k=0

    #     # plot original data
    #     fig_u = plt.figure()
    #     ax_u = Axes3D(fig_u)
    #     ax_u.plot_surface(x_free.reshape(ny-2,nx-2), y_free.reshape(ny-2,nx-2), orig_data_u[k].reshape(ny-2,nx-2),cmap=cm.viridis, rstride=1, cstride=1)
    #     ax_u.view_init(elev=30,azim=30)
    #     # ax_u = fig_u.gca()
    #     # p_u=ax_u.pcolor(x_free.reshape(ny-2,nx-2), y_free.reshape(ny-2,nx-2), orig_data_u[k].reshape(ny-2,nx-2))
    #     # cb_u=fig_u.colorbar(p_u,ax=ax_u)
    #     ax_u.set_xlabel('$x_{free}$')
    #     ax_u.set_ylabel('$y_{free}$')
    #     plt.title('Original $u$')
    #     plt.show()

    #     fig_v = plt.figure()
    #     ax_v = Axes3D(fig_v)
    #     ax_v.plot_surface(x_free.reshape(ny-2,nx-2), y_free.reshape(ny-2,nx-2), orig_data_v[k].reshape(ny-2,nx-2),cmap=cm.viridis, rstride=1, cstride=1)
    #     ax_v.view_init(elev=30,azim=60)
    #     # ax_v = fig_v.gca()
    #     # p_v=ax_v.pcolor(x_free.reshape(ny-2,nx-2), y_free.reshape(ny-2,nx-2), orig_data_v[k].reshape(ny-2,nx-2))
    #     # cb_v=fig_v.colorbar(p_v,ax=ax_v)
    #     ax_v.set_xlabel('$x_{free}$')
    #     ax_v.set_ylabel('$y_{free}$')
    #     plt.title('Original $v$')
    #     plt.show()


    # In[25]:


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

    # set dataset
    dataset_u = {'train':data_utils.TensorDataset(torch.tensor(trainset_u,dtype=torch.float32)),
               'test':data_utils.TensorDataset(torch.tensor(testset_u,dtype=torch.float32))}
    dataset_v = {'train':data_utils.TensorDataset(torch.tensor(trainset_v,dtype=torch.float32)),
               'test':data_utils.TensorDataset(torch.tensor(testset_v,dtype=torch.float32))}

    print(dataset_u['train'].tensors[0].shape, dataset_u['test'].tensors[0].shape)
    print(dataset_v['train'].tensors[0].shape, dataset_v['test'].tensors[0].shape)


    # In[26]:


    # compute dataset shapes
    dataset_u_shapes = {'train':trainset_u.shape,
                        'test':testset_u.shape}
    dataset_v_shapes = {'train':trainset_v.shape,
                        'test':testset_v.shape}

    print(dataset_u_shapes['train'],dataset_u_shapes['test'])
    print(dataset_v_shapes['train'],dataset_v_shapes['test'])


    # In[27]:


    # set data loaders
    train_loader_u = DataLoader(dataset=dataset_u['train'], 
                              batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader_v = DataLoader(dataset=dataset_v['train'], 
                              batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_u = DataLoader(dataset=dataset_u['test'], 
                             batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_v = DataLoader(dataset=dataset_v['test'], 
                             batch_size=batch_size, shuffle=True, num_workers=0)

    data_loaders_u = {'train':train_loader_u, 'test':test_loader_u}
    data_loaders_v = {'train':train_loader_v, 'test':test_loader_v}


    # In[28]:


    # set the number of nodes in each layer
    a = 2
    b = int(100)
    db = int(10)

    M1 = int(a*m) # encoder hidden layer
    M2 = b + (m-1)*db # decoder hidden layer

    f = LS_dim # latent dimension

    # sparsity and shape of mask
    mask_2d=create_mask_2d(m,b,db)

    # number of parameters and memory
    en_para=m*M1+M1+M1*f
    de_para=f*M2+M2+np.count_nonzero(mask_2d)
    print('Encoder parameters:{:.8e}({:.4}GB)'.format(en_para,en_para*4/2**30),      'Decoder parameters:{:.8e}({:.4}GB)'.format(de_para,(f*M2+M2+M2*m)*4/2**30))

    # data size
    data_size=np.prod(orig_data_u.shape)
    print('Data size:{:.8e}({:.4}GB)'.format(data_size,data_size*4/2**30))


    # In[29]:


    # Set file names
    if Re==10000:
        file_name_AE_u= '../../../../../usr/WS1/fries4/Experiments/2DBurgers/model/AE_u_{}_dim_{}.tar'.format(25, LS_dim)
        file_name_AE_v= '../../../../../usr/WS1/fries4/Experiments/2DBurgers/model/AE_v_{}_dim_{}.tar'.format(25, LS_dim)
        file_name_AE="./model/AE_high_Re_v3_batch_240.p"

        PATH_u = './checkpoint_u_high_Re_v3_batch_240.tar'
        PATH_v = './checkpoint_v_high_Re_v3_batch_240.tar'
    elif Re==100:
        file_name_AE_u="./model/AE_u_low_Re_v3_batch_240.pkl"
        file_name_AE_v="./model/AE_v_low_Re_v3_batch_240.pkl"
        file_name_AE="./model/AE_low_Re_v_3_batch_240.p"

        PATH_u = './checkpoint_u_low_Re_v3_batch_240.tar'
        PATH_v = './checkpoint_v_low_Re_v3_batch_240.tar'
    else:
        raise NameError('{} is given for Re, but it must be either 100 or 10000'.format(Re))  


    # In[30]:


    # release gpu memory
    torch.cuda.empty_cache()


    # ## For u

    # In[31]:


    # load model
    try:
        checkpoint = torch.load(PATH_u, map_location=device)

        encoder = Encoder(m,M1,f).to(device)
        decoder = Decoder(f,M2,m).to(device)

        # Prune
        prune.custom_from_mask(decoder.full[2], name='weight', mask=torch.tensor(mask_2d).to(device))    

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

        # resume training
        print("")
        print('Re-start {}th training... m={}, f={},a={}, b={}, db={}'.format(
            last_epoch+1, m, f, a, b, db))
    except:
        encoder = Encoder(m,M1,f).to(device)
        decoder = Decoder(f,M2,m).to(device)

        # Prune
        prune.custom_from_mask(decoder.full[2], name='weight', mask=torch.tensor(mask_2d).to(device))

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

        # start training
        print("")
        print('Start first training... m={}, f={}, a={}, b={}, db={}'          .format(m, f, a, b, db))
    pass


    # In[32]:


    # train model
    since = time.time()

    for epoch in range(last_epoch+1,num_epochs+1):   

        if epoch%num_epochs_print == 0:
            print()
            print('Epoch {}/{}, Learning rate {}'.format(
                epoch, num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
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
            for data, in data_loaders_u[phase]:
                inputs = data.to(device)
                targets = data.to(device)

                if phase == 'train':
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
                    with torch.set_grad_enabled(False):
                        outputs = decoder(encoder(inputs))
                        running_loss += loss_func(outputs,targets).item()*inputs.shape[0]

            # compute epoch loss
            epoch_loss = running_loss / dataset_u_shapes[phase][0]
            loss_hist[phase].append(epoch_loss)

            # update learning rate
            if phase == 'train':
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
                        }, PATH_u)        

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
    encoder.to('cpu').eval()
    decoder.to('cpu').eval()

    with torch.set_grad_enabled(False):
        train_inputs = torch.tensor(data_u[train_ind])
        train_targets = torch.tensor(data_u[train_ind])
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

    # save models
    print()
    print("Saving after {}th training to".format(epoch),file_name_AE_u)
    encoder_u = encoder
    decoder_u = decoder
    torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, '../../../../../usr/WS1/fries4/Experiments/2DBurgers/model/AE_u.tar')
    #torch.save((encoder,decoder),file_name_AE_u)


    # In[33]:


    # delete checkpoint
    try:
        os.remove(PATH_u)
        print()
        print("checkpoint removed")
    except:
        print("no checkpoint exists") 


    # In[34]:


    # plot train and test loss
    plt.figure()
    plt.semilogy(loss_hist['train'])
    plt.semilogy(loss_hist['test'])
    plt.legend(['train','test']) 


    # In[35]:


    # release gpu memory
    torch.cuda.empty_cache()


    # ## For v

    # In[36]:


    # load model
    try:
        checkpoint = torch.load(PATH_v, map_location=device)

        encoder = Encoder(m,M1,f).to(device)
        decoder = Decoder(f,M2,m).to(device)

        # Prune
        prune.custom_from_mask(decoder.full[2], name='weight', mask=torch.tensor(mask_2d).to(device))    

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

        # resume training
        print("")
        print('Re-start {}th training... m={}, f={}, a={}, b={}, db={}'.format(
            last_epoch+1, m, f, a, b, db))
    except:
        encoder = Encoder(m,M1,f).to(device)
        decoder = Decoder(f,M2,m).to(device)

        # Prune
        prune.custom_from_mask(decoder.full[2], name='weight', mask=torch.tensor(mask_2d).to(device))

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

        # start training
        print("")
        print('Start first training... m={}, f={}, a={}, b={}, db={}'.format(
            m, f, a, b, db))
    pass


    # In[37]:


    # train model
    since = time.time()

    for epoch in range(last_epoch+1,num_epochs+1):   

        if epoch%num_epochs_print == 0:
            print()
            print('Epoch {}/{}, Learning rate {}'.format(
                epoch, num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
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
            for data, in data_loaders_v[phase]:
                inputs = data.to(device)
                targets = data.to(device)

                if phase == 'train':
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
                    with torch.set_grad_enabled(False):
                        outputs = decoder(encoder(inputs))
                        running_loss += loss_func(outputs,targets).item()*inputs.shape[0]

            # compute epoch loss
            epoch_loss = running_loss / dataset_v_shapes[phase][0]
            loss_hist[phase].append(epoch_loss)

            # update learning rate
            if phase == 'train':
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
                        }, PATH_v)        

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
    encoder.to('cpu').eval()
    decoder.to('cpu').eval()

    with torch.set_grad_enabled(False):
        train_inputs = torch.tensor(data_v[train_ind])
        train_targets = torch.tensor(data_v[train_ind])
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

    # save models
    print()
    print("Saving after {}th training to".format(epoch),file_name_AE_v)
    encoder_v = encoder
    decoder_v = decoder
    #torch.save((encoder,decoder),file_name_AE_v)
    torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, '../../../../../usr/WS1/fries4/Experiments/2DBurgers/model/AE_v.tar')

    # In[38]:


    # delete checkpoint
    try:
        os.remove(PATH_v)
        print()
        print("checkpoint removed")
    except:
        print("no checkpoint exists") 


    # In[39]:


    # # plot train and test loss
    # plt.figure()
    # plt.semilogy(loss_hist['train'])
    # plt.semilogy(loss_hist['test'])
    # plt.legend(['train','test'])
    #     plt.show()   


    # In[40]:


    torch.cuda.empty_cache()




