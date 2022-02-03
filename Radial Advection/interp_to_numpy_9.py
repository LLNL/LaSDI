#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:45:03 2021

@author: friesingcold
"""

import numpy as np
import mfem.ser as mfem
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import path
from tqdm import tqdm
from mfem import path
from os.path import expanduser, join, dirname, exists
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

path = dirname(dirname(__file__))
meshfile = expanduser(join(path, 'ex9_sim','Example9_000000','mesh.000000'))   
mesh = mfem.Mesh(meshfile)

x = np.linspace(-1,1,512)
y = np.linspace(-1,1,512)
X,Y = np.meshgrid(x,y, indexing = 'xy')
points = np.column_stack([X.ravel(), Y.ravel()])
interp = mesh.FindPoints(points)
intpoints = interp[2].ToList()

t_steps = np.arange(0,1200,5)
index = -1
final = np.empty([len(t_steps), points.shape[0]])
for t in tqdm(t_steps):
    index += 1
    gf_file = 'ex9_sim/Example9_{}/solution.000000'.format(str(t).zfill(6))
    u = mfem.GridFunction(mesh, gf_file)
    for i in range(points.shape[0]):
        final[index,i]=u.GetValue(interp[1][i], intpoints[i])
    
import pickle

final = final.reshape(-1,512,512)[:,::8,::8]
final = final.reshape(-1,64**2).astype('float32')        

np.savez_compressed('ex9_interp_{}'.format(sys.argv[1]), final)
