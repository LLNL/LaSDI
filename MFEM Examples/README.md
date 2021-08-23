# DIALS MFEM Examples
Note: Files names will differ and will need to be adjusted

## Instructions

1) Generate MFEM files using "visit" option in generation. 

2) Use "interp_to_numpy.py" to transfer from FEM to Finite difference method
 
Note, that for consistency, the interpolation must remain the same across all training and testing values. The autoencoder training and DIALS code is set up to use 
512x512 (and reduced to 64x64 for computational purposes). If you use 256x256 interpolation and reduce this to 64x64, the results will be incosistent with 512x512 reduced to 64x64.

3) Use the "File_compiler.ipynb" to put desired snapshots together into one file.

3) Train using same autoencoder as 2d Burgers ("Train_MFEM_EX9.py")

4) Apply DIALS

### Questions/Comments
Questions and comments should be directed to frieswd@math.arizona.edu

