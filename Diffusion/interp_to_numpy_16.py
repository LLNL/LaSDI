#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:45:03 2021

This file uses PyMFEM to convert files from FE data to that on a 64x64 fixed mesh grid. First, we interpolate on 512x512 grid and then reduce this to 64x64.

Last Updated by Bill Fries 2/2/22
"""

import numpy as np
import mfem.ser as mfem
import matplotlib.pyplot as plt
import path
from tqdm import tqdm
from mfem import path
from os.path import expanduser, join, dirname, exists
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

path = dirname(dirname(__file__))
meshfile = expanduser(join(path,'Example16_000000','mesh.000000'))   
mesh = mfem.Mesh(meshfile)

x = np.linspace(0,1,512)
y = np.linspace(0,1,512)
X,Y = np.meshgrid(x,y, indexing = 'xy')
points = np.column_stack([X.ravel(), Y.ravel()])
interp = mesh.FindPoints(points)
intpoints = interp[2].ToList()

t_steps = np.arange(0,100)
index = -1
final = np.empty([len(t_steps), points.shape[0]])
for t in tqdm(t_steps):
    index += 1
    gf_file = 'Example16_{}/temperature.000000'.format(str(t).zfill(6))
    u = mfem.GridFunction(mesh, gf_file)
    for i in range(points.shape[0]):
        final[index,i]=u.GetValue(interp[1][i], intpoints[i])
    
final = final.reshape(-1,512,512)[:,::8,::8]
final = final.reshape(-1,64**2).astype('float32')        

np.savez_compressed('./data/ex16_w_{}_a_{}'.format(sys.argv[1], sys.argv[2]), final)
