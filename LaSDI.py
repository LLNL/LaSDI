import numpy as np
import numpy.linalg as LA
import pysindy as ps
import matplotlib.pyplot as plt

class LaSDI:
    """
    LaSDI class for data-driven ROM. Functions: train_dynamics approximates dynamical systems of the latent-space. 
                                                generate_FOM uses an initial condition and parameter values to generate a new model
    NOTE: To avoid errors, make sure to set NN = True for use with autoencoder.
    
    Inputs:
       encoder: either neural network (with pytorch) or matrix (LS-ROM)
       decoder: either neural network (with pytorch) or matrix (LS-ROM)
       NN: Boolean on whether a NN is used
       device: device NN is on. Default 'cpu', use 'cuda' if necessary
       Local: Boolean. Determines Local or Global DI (still in progress)
       Coef_interp: Boolean. Determines method of Local DI
       nearest_neigh: Number of nearest neigh in Local DI
    """
    
    def __init__(self, encoder, decoder, NN = False, device = 'cpu', Local = False, Coef_interp = False, nearest_neigh = 4):
        
        self.Local = Local
        self.Coef_interp = Coef_interp
        self.nearest_neigh = nearest_neigh
        self.NN = NN
        if NN == False:
            self.IC_gen = lambda params: np.matmul(encoder, params)
            self.decoder = lambda traj: np.matmul(decoder, traj.T)
            
        else:
            import torch
            self.IC_gen = lambda IC: encoder(torch.tensor(IC).to(device)).cpu().detach().numpy()
            self.decoder = lambda traj: decoder(torch.tensor(traj.astype('float32')).to(device)).cpu().detach().numpy()
            
        return
    
    def train_dynamics(self,ls_trajs, training_values, dt, normal = 1, degree = 1, include_interaction=False, LS_vis = True ):
        """
        Approximates the dynamical system for the latent-space. Local == True, use generate_FOM. 
        
        Inputs:
           ls_trajs: latent-space trajectories in a list of arrays formatted as [time, space]
           training_values: list/array of corresponding parameter values to above
           dt: time-step used in FOM
           normal: normalization constant. Default as 1, set 'max' to normalize all values to between -1 and 1
           LS_vis: Boolean to visulaize a trajectory and discovered dynamics in the latent-space. Default True
           
           PySINDy parameters:
              degree: degree of desired polynomial. Default 1
              include_interactions: Boolean include cross terms for degree >1. Default False
        """
        if normal == "max":
            self.normal = np.amax(np.abs(ls_trajs))
        else:
            self.normal = normal
        
        data_LS = []
        for traj in ls_trajs:
            data_LS.append(traj/normal)
            
        poly_library = ps.PolynomialLibrary(include_interaction=include_interaction, degree = degree)
        optimizer = ps.STLSQ(alpha=0, copy_X=True, fit_intercept=False, max_iter=20, normalize=False, ridge_kw=None, threshold=0)
        if self.Local == False:
            model = ps.SINDy(feature_library = poly_library, optimizer = optimizer)
            model.fit(data_LS, t = dt, multiple_trajectories = True)
            self.model = model
            if LS_vis == True:
                fig = plt.figure()
                ax = plt.axes()
                labels = {'orig': 'Latent-Space Trajectory', 'new': 'Approximated Dynamics'}
                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(data_LS[-1][:-1,dim], alpha = .5, label = labels['orig'])
                    labels['orig'] = '_nolegend_'
                plt.gca().set_prop_cycle(None)
                new = model.simulate(data_LS[-1][0], np.linspace(0, 1, len(data_LS[-1])))
                for dim in range(data_LS[-1].shape[1]):
                    plt.plot(new[:,dim], '--', label = labels['new'])
                    labels['new'] = '_nolegend_'
                ax.legend()
                plt.show()
            return model.print()  

        else:
            self.ls_trajs = ls_trajs
            self.training_values = training_values
            self.dt = dt
            self.degree = degree
            self.include_interaction = include_interaction
            self.data_LS = data_LS
            self.poly_library = poly_library
            self.optimizer = optimizer
            
#             return print('Warning: This module is not completed yet. Use LaSDI.generate_FOM')
            return
        """
        Work in Progress (Partitioning parameter space and training one model for each)

            
        if Local == True:
            
            if Coef_interp = False:
                model_list = []
        """        
        
            
    
    def generate_FOM(self,pred_IC,pred_value,t):
        """
        Takes initial condition in full-space and associated parameter values and generates forward in time using the trained dynamics from above.
        Inputs:
            pred_IC: Initial condition of the desired simulation
            pred_value: Associated parameter values
            t: time stamps corresponding to training FOMs
        """
        IC = self.IC_gen(pred_IC)
        if self.Local == False:
            latent_space_recon = self.normal*self.model.simulate(IC, t)
            FOM_recon = self.decoder(latent_space_recon)
            if self.NN == False:
                return FOM_recon.T
            else:
                return FOM_recon
        
        else:
            if self.Coef_interp == False:
                dist = np.empty(len(self.training_values))
                for iii,P in enumerate(self.training_values):
                    dist[iii]=(LA.norm(P-pred_value))

                k = self.nearest_neigh
                dist_index = np.argsort(dist)[0:k]
                local = []
                j=-1
                for iii in dist_index:
                    local.append(self.data_LS[iii])
                model = ps.SINDy(feature_library = self.poly_library, optimizer = self.optimizer)    
                model.fit(local, t = self.dt, multiple_trajectories = True, quiet = True)
                
                latent_space_recon = self.normal*model.simulate(IC, t)
                FOM_recon = self.decoder(latent_space_recon)
                
                return FOM_recon.T
        
        