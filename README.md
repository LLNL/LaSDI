# DIALS
## Disclaimer:
For 1D and 2D Burgers, file organization is still ongoing. I will be modifying these files to (a) make the file locations easily modifiable and consistent (b) make my previous simulations accessible when working on LASSEN (c) make one file that builds, trains, and applies DIALS in one go.
## Orgainization

There are three examples included. 1) 1D Burgers, 2) 2D Burgers (both as simulated in [https://arxiv.org/abs/2009.11990]) and 3) a radial advection example as from MFEM (example 9, problem 3)
Each of the folders includes instructions for building the data, training the neural networks and applying DIALS.
They also include basic data files and trained networks for observing some of the results. 

## Instructions

(Check each script for file locations. If running on Lassen, you can access my files for interpolations/models. For producing your own results, make sure the file names are consistent)

To generate results, run Build_<*>.ipynb (1D/2D Burgers) or import VISIT files from various MFEM examples and train models using Train_<*>.ipynb. These results might not be the most accurate because of the lack of lots of training data. If you wish to consider more data but do not want to build/train the models, contact me for the file locations.
*Note the "..._local_compare.ipynb" files use more testing points. Contact me if you have trouble getting to the file locations.



If you wish to compile your own results, modify the Build_<*>.ipynb (1D/2D Burgers) or import VISIT files from various MFEM examples. 

In the case of MFEM examples, it is necessary to transfer the simultion from finite elements to finite difference. This is completed by the Interp_MFEM.ipynb file.
Note, that for consistency, the interpolation must remain the same across all training and testing values. The autoencoder training and DIALS code is set up to use 
512x512 (and reduced to 64x64 for computational purposes). If you use 256x256 interpolation and reduce this to 64x64, the results will be incosistent with 512x512 reduced to 64x64.



## Notes on Training Autoencoders and Applying DIALS

Below is to help intution on modifying the code as necessary:

### Training AE:

Various snapshots, need to retain differences in initial conditions. The easiest method to do this is to regularize so that max_(all snapshots & all time points & all space points) = 1. 
If the generated data already fits within this regime, then do not modify the snapshots when training the network. 

### Applying DIALS:

First Pass: Try to fit either degree = 1 or degree = 2 (with "include_interactions = FALSE then = True")
	Visually verify the fit and through MSE in latent-space. 
Second Pass: If above does not work, normalize the latent-space data by dividing by the max(abs()) over all snapshots in latent-space. (It's important that if you modify one snapshot, then you modify all snapshots in the same way). 
	This method requires you to multiply by the normalization factor after applying the ODE integrator. 
Third Pass: You can increase the degree with and without interactions as necessary. However, this makes the integration much more unstable in some situations. Proceed with Caution
Fourth Pass: If the above does not work, then contact me for further and more complex techniques (such as appending the latent-space). 

### Questions/Comments
Questions and comments should be directed to frieswd@math.arizona.edu

