# DIALS
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

### Questions/Comments
Questions and comments should be directed to frieswd@math.arizona.edu

