#!/bin/bash
#SBATCH -N 1
#SBATCH -J bash_ex9
#SBATCH -t 2:00:00
#SBATCH -p pbatch
#SBATCH -o bash_ex9.log
#SBATCH --open-mode truncate

source ../lasdi_venv/bin/activate

MFEM_DIR="../dependencies/mfem/examples"

cp lasdi_ex16.cpp $MFEM_DIR
cp makefile_ex16 $MFEM_DIR
cp interp_to_numpy_16.py $MFEM_DIR

cd $MFEM_DIR
rm ex16_interp_*.npz
make lasdi_ex16 -f makefile_ex16

for i in $(seq 180 20 1220)
do
	for j in $(seq 180 20 220)
	do
		echo $i$j
		./lasdi_ex16 -m ../data/inline-tri.mesh -vs 1 -r 3 -visit -freq $((i/100)).$((i%100)) -am $((j/100)).$((j%100)) -tf 1
	    python interp_to_numpy_16.py $i $j
	done
done

make clean
cd ../../../Diffusion/
rm -rf data
mkdir data
mv -f $MFEM_DIR/ex16_interp_*.npz ./data/
