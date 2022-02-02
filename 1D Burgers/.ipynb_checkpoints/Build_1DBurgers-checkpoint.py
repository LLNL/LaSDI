#!/usr/bin/env python
# coding: utf-8

"""
This code is modified from the code found in https://arxiv.org/abs/2011.07727. This generates a training snapshot called "snapshot_git.p" with parameters values as printed. 
It also generates a FOM for comparison saved as "FOM.p". This is also done at the parameter value listed when running the script.

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

from scipy import sparse as sp
from scipy import sparse
from scipy.sparse import spdiags
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
from scipy.io import savemat,loadmat
import scipy.integrate as integrate

import matplotlib.pyplot as plt

import sys,time
import os
import copy
import pickle

maxk = 10
convergence_threshold = 1.0e-8

nx = 1001
dx = 6 / (nx - 1)
nt = 1000
tstop = 1
x=np.linspace(-3, 3, nx)

dt = tstop / nt 
c = dt/dx
t = np.linspace(0, tstop, nt)
# ## Functions


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
    
def sine_wave(amp,width):
    
    u0 = np.zeros(nx)
    u0[1:int(width/dx+1)] =amp/2*(np.sin(2*np.pi/(x[int(width/dx+1)]-x[1])*x[1:int(width/dx+1)]-np.pi/2)+1)
    u0[-1] = u0[0]
    
    return u0


def gaussian(amp,width):
    
    u0 = amp*np.exp(-(x-0.0)**2/(2*width**2))
    u0[-1] = u0[0]
    
    return u0

def residual(un,uw,c,idxn1):
    
    # r = -u^{n} + u^{n+1} -dt*f(u^{n+1})
    
    f = c*(uw**2 - uw*uw[idxn1]) 
    
    r = -un + uw + f
    
    return r

def jacobian(u,c,idxn1):

    # J = I - dt*dfdu
    
    diag_comp = 1.0 + c*(2*u - u[idxn1])
    subdiag_comp = np.ones(nx-1)
    subdiag_comp[:-1] = -c*u[1:]
        
    data = np.array([diag_comp, subdiag_comp])
    J = spdiags(data,[0,-1],nx-1,nx-1,format='csr')
    J[0,-1] = -c*u[0]
    
    return J

def solve(u0):

    u = np.zeros((nt+1,nx))
    u_inter=np.array([])
    u[0] = u0
    u_inter=np.append(u_inter,u0[:-1])
    I = sparse.eye(nx,format='csr')
    for n in range(nt): 
        uw = u[n,:-1].copy()
        r = residual(u[n,:-1],uw,c,idxn1)
        
        for k in range(maxk):
            J = jacobian(uw,c,idxn1)
            duw = spsolve(J, -r)
#             duw = np.linalg.solve(J,-r)
            uw = uw + duw
            r = residual(u[n,:-1],uw,c,idxn1)
            u_inter=np.append(u_inter,uw)

            rel_residual = np.linalg.norm(r)/np.linalg.norm(u[n,:-1])
            if rel_residual < convergence_threshold:
                u[n+1,:-1] = uw.copy()
                u[n+1,-1] = u[n+1,0]
                break
    
    return u,u_inter.reshape((-1,nx-1))

def generate_dataset(amp_arr,width_arr):
    
    num_amp=amp_arr.shape[0]
    num_width=width_arr.shape[0]
    data = []
    data_inter = []
    for i in range(num_amp):
        for j in range(num_width):
            u0=gaussian(amp_arr[i],width_arr[j])
#             u0=sine_wave(amp_arr[i],width_arr[j])
            u,u_inter=solve(u0)
            data.append(u)
            data_inter.append(u_inter)
    data = np.vstack(data)   
    data_inter = np.vstack(data_inter)   
    
    return data, data_inter

dn1=kth_diag_indices(np.eye(nx-1),-1)
idxn1=np.zeros(nx-1,dtype='int')
idxn1[1:]=np.arange(nx-2)
idxn1[0]=nx-2


amp_arr = np.array([.75,.85])
width_arr = np.array([.95,1.05])
print("Amplitudes:", amp_arr)
print("Widths:", width_arr)

snapshot_full,snapshot_full_inter = generate_dataset(amp_arr,width_arr)
pickle.dump(snapshot_full.astype('float32'), open("./data/snapshot_git.p", "wb"))

# print(snapshot_full.shape)
# print(snapshot_full_inter.shape)

print('')
print('Full Order Model:')
amp_arr_FOM = np.array([0.8])
width_arr_FOM = np.array([1.0])
print("Amplitudes:", amp_arr_FOM)
print("Widths:", width_arr_FOM)

FOM_start = time.time()
snapshot_full_FOM,snapshot_full_inter_FOM = generate_dataset(amp_arr_FOM,width_arr_FOM)
FOM_time = time.time()-FOM_start
pickle.dump({'FOM': snapshot_full_FOM.astype('float32'), 'time': FOM_time}, open("./data/FOM.p", "wb"))


