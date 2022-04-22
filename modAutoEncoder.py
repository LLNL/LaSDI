import os
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
from torch.optim import lr_scheduler

from scipy import sparse as sp
from scipy import sparse
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
from scipy.io import savemat,loadmat
import scipy.integrate as integrate

from itertools import combinations_with_replacement, product
from sklearn.decomposition import SparseCoder
import copy

def silu(input):
  return input * torch.sigmoid(input)

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input):
    return silu(input)
    
class Encoder(nn.Module):
  def __init__(self,m,M1,f,f_a):
    self.m = m
    self.f = f
    self.M = M1
    super(Encoder,self).__init__()
    self.full = nn.Sequential(  nn.Linear(m,M1),
                                f_a(),
                                nn.Linear(M1,f,bias=False) )
              
  def forward(self, y):     
    y = y.view(-1,self.m)
    T = self.full(y)
    T = T.squeeze()
    
    return T
    
class Decoder(nn.Module):
  def __init__(self,f,M2,m,f_a):
    self.m = m
    self.f = f
    self.M = M2
    super(Decoder,self).__init__()
    self.full = nn.Sequential(  nn.Linear(f,M2,bias=False),
                                f_a(),
                                nn.Linear(M2,m,bias=False) )
           
  def forward(self,T):
    T = T.view(-1,self.f)
    y = self.full(T)
    y = y.squeeze()
    
    return y

def getDevice():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return device

def createAE(  EncoderClass,
               DecoderClass,
               f_activation,
               mask,
               m, f, M1, M2,
               device ):
  encoder = EncoderClass(m,M1,f,f_activation).to(device)
  decoder = DecoderClass(f,M2,m,f_activation).to(device)
  # Prune
  prune.custom_from_mask(decoder.full[2], name='weight', mask=torch.tensor(mask).to(device))    
  return encoder, decoder

def readAEFromFile( EncoderClass,
                    DecoderClass,
                    f_activation,
                    mask,
                    m, f, M1, M2,
                    device,
                    fname ):
  encoder, decoder = createAE(  EncoderClass,
                                DecoderClass,
                                f_activation,
                                mask,
                                m, f, M1, M2,
                                device )
  model = torch.load(fname, map_location=device)
  encoder.load_state_dict(model['encoder_state_dict'])
  decoder.load_state_dict(model['decoder_state_dict'])
  return encoder, decoder

def trainAE( encoder, 
             decoder,
             training_data,
             test_data,
             batch_size,
             num_epochs,
             num_epochs_print,
             early_stop_patience,
             model_fname,
             chkpt_fname,
             plt_fname = 'training_loss.png',
             num_epochs_save_model = 9999999 ):

  dataset = {'train':data_utils.TensorDataset(torch.tensor(training_data)),
             'test':data_utils.TensorDataset(torch.tensor(test_data))}
  dataset_shapes = {'train':training_data.shape, 'test':test_data.shape}

  # set data loaders
  train_loader = DataLoader(dataset=dataset['train'],
                            batch_size=batch_size, shuffle=True, num_workers=0)
  test_loader = DataLoader(dataset=dataset['test'],
                           batch_size=batch_size, shuffle=True, num_workers=0)
  data_loaders = {'train':train_loader, 'test':test_loader}
  
  # set device
  device = getDevice()

  # load model
  try:
      checkpoint = torch.load(chkpt_fname, map_location=device)
      
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
      print('Re-start {}th training... m={}, f={}, M1={}, M2={}'.format(
            last_epoch+1, encoder.m, encoder.f, encoder.M, decoder.M))
  except:
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
      print('Start first training... m={}, f={}, M1={}, M2={}'.format(
            encoder.m, encoder.f, encoder.M, decoder.M))
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
                      }, chkpt_fname)
  
      if epoch%num_epochs_save_model==0:
          print("Saving after {}th training to".format(epoch),
                model_fname )
          torch.save( { 'encoder_state_dict': encoder.state_dict(), 
                        'decoder_state_dict': decoder.state_dict()}, 
                      model_fname )
          # plot train and test loss
          plt.figure()
          plt.semilogy(loss_hist['train'])
          plt.semilogy(loss_hist['test'])
          plt.legend(['train','test'])
          plt.savefig(plt_fname)
  
  
  print()
  print('Epoch {}/{}, Learning rate {}'.format(epoch, num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
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
      train_inputs = torch.tensor(training_data).to(device)
      train_targets = torch.tensor(training_data).to(device)
      train_outputs = decoder(encoder(train_inputs))
      train_loss = loss_func(train_outputs,train_targets).item()
  
  # print out training time and best results
  print()
  if epoch < num_epochs:
      print('Early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'.format(epoch-last_epoch, time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
  else:
      print('No early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'.format(epoch-last_epoch, time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
  print('-' * 10)
  print('Best train MSELoss: {}'.format(train_loss))
  print('Best test MSELoss: {}'.format(best_loss))
  
  ###### save models ########
  print()
  print("Saving after {}th training to".format(epoch),
        model_fname)
  torch.save( {'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, 
              model_fname )
  
  
  # plot train and test loss
  plt.figure()
  plt.semilogy(loss_hist['train'])
  plt.semilogy(loss_hist['test'])
  plt.legend(['train','test'])
  #plt.show()   
  plt.savefig(plt_fname)
  
  # delete checkpoint
  try:
      os.remove(chkpt_fname)
      print()
      print("checkpoint removed")
  except:
      print("no checkpoint exists")
      
  torch.cuda.empty_cache()

def encodedSnapshots( encoder,
                      solution_snapshots,
                      n_steps,
                      device ):
  ndata = solution_snapshots.shape[0]
  nset = int(ndata/n_steps)
  latent_space_SS = []
  for i in range(nset):
    input_SS = torch.tensor(solution_snapshots[i*n_steps:(i+1)*n_steps]).to(device)
    latent_space = encoder(input_SS).cpu().detach().numpy()
    latent_space_SS.append(latent_space)

  return latent_space_SS
