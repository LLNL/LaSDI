# LaSDI
## Orgainization

There are four examples included. 1) 1D Burgers, 2) 2D Burgers (both as simulated in [https://arxiv.org/abs/2009.11990]), 3) a radial advection example as from MFEM (example 9, problem 3) and 4) a time-dependent diffusion example (MFEM example 16)
Each of the folders includes instructions for building the data, training the neural networks or using a linear data-compression technique and applying LaSDI.
They also include basic data files and trained networks for observing some of the results. 

## Instructions

### 1D and 2D Burger's Problem

To generate results, first generate the training data using the "Build" file. For LaSDI-NM you can then train an auto-encoder using the "Train" file. Finally, use the "LaSDI" files to generate the ROM. Note that for LaSDI-LS, only the training files are required, as the POD data-compression is done within the LaSDI-LS notebook.

### MFEM Examples

Because the MFEM data comes from finite-element data, we first translate the FE degrees of freedom to a fixed 64x64 grid. This can be done using the "bash" file for both the Diffusion and Radial Advection examples. This is followed by the "Train" and "LaSDI" files as in the Burger's examples. Further details can be found in the problem folders.

## Notes on Training Autoencoders and Applying LaSDI

Below is to help intution on modifying the code as necessary:

### Training AE:

Various snapshots, need to retain differences in initial conditions. The easiest method to do this is to regularize so that max_(all snapshots & all time points & all space points) = 1. 
If the generated data already fits within this regime, then do not modify the snapshots when training the network. 

### Applying LaSDI:

The LaSDI class is documented with inputs, outputs and general instructions. Various *kwargs* can be passed through to adjust the learning process. In general:

1. LaSDI(

        Inputs:
           encoder: either neural network (with pytorch) or matrix (LS-ROM)
           decoder: either neural network (with pytorch) or matrix (LS-ROM)
           NN: Boolean on whether a nerual network is used
           device: device NN is on. Default 'cpu', use 'cuda' if necessary
           Local: Boolean. Determines Local or Global DI
           Coef_interp: Boolean. Determines method of Local DI
           nearest_neigh: Number of nearest neigh in Local DI
	   Coef_interp_method: Either Rbf or interp2d (method for coefficient interpolation)
           
       
2. LaSDI.train_dynamics(

        Inputs:
           ls_trajs: latent-space trajectories in a list of arrays formatted as [time, space] *Currently working on implementation to generate ls_trajectories within the method*
           training_values: list/array of corresponding parameter values to above
           dt: time-step used in FOM
           normal: normalization constant to scale the magnitude of the latent-space trajectories. Ideally, trajectory magnitude is between -1 and 1. Default as 1.
           LS_vis: Boolean to visulaize a trajectory and discovered dynamics in the latent-space. Default True.
           
           PySINDy parameters:
              degree: degree of desired polynomial. Default 1
              include_interactions: Boolean include cross terms for degree >1. Default False
	      threshold: Sparsity threshold for high-degree approximations to encourage numerical stability.
           
 
3. LaSDI.generate_FOM(


        Inputs:
            pred_IC: Initial condition of the desired simulation
            pred_value: Associated parameter values
            t: time stamps corresponding to training FOMs
            

First Pass: Try to fit either degree = 1 or degree = 2 (with "include_interactions = FALSE then = True"). Visually verify the fit and through MSE in latent-space. 

Second Pass: If above does not work, normalize the latent-space data by dividing by the max(abs()) over all snapshots in latent-space. (It's important that if you modify one snapshot, then you modify all snapshots in the same way). This method requires you to multiply by the normalization factor after applying the ODE integrator. 

Third Pass: You can increase the degree with and without interactions as necessary. However, this makes the integration much more unstable in some situations. Proceed with Caution

Fourth Pass: If the above does not work, then contact me for further and more complex techniques (such as appending the latent-space). 

### Questions/Comments
Questions and comments should be directed to frieswd@math.arizona.edu

