#!/bin/bash
#SBATCH -N 1
#SBATCH -J bash_ex9
#SBATCH -t 2:00:00
#SBATCH -p pbatch
#SBATCH -o bash_ex9.log
#SBATCH --open-mode truncate

source ../lasdi_venv/bin/activate

MFEM_DIR="../dependencies/mfem/examples"

cp lasdi_ex9.cpp $MFEM_DIR
cp makefile_ex9 "$MFEM_DIR"
cp interp_to_numpy_9.py "$MFEM_DIR"

cd "$MFEM_DIR"
rm ex9_interp_*.npz
rm -rf ex9_sim
mkdir ex9_sim
make lasdi_ex9 -f makefile_ex9

for i in $(seq 95 5 105)
do
	echo $i
        rm -rf ex9_sim/* 
	./lasdi_ex9 -m ../data/periodic-square.mesh -p 3 -r 3 -tf 3 -dt .0025 -vs 5 -freq $((i/100)).0$((i%100)) -visit
        mv -f Example9* ex9_sim
        mv -f ex9*gf ex9_sim
	python interp_to_numpy_9.py $i
done

make clean
cd ../../../Radial\ Advection/
rm -rf data
mkdir data
mv -f $MFEM_DIR/ex9_interp_*.npz ./data/
