#!/bin/bash

for i in $(seq 180 20 220)
do
	for j in $(seq 180 20 220)
	do
		echo $i$j
		./ex16 -m ../data/inline-tri.mesh -vs 1 -r 3 -visit -freq $((i/100)).$((i%100)) -am $((j/100)).$((j%100)) -tf 1
		python interp_to_numpy.py $i $j
	done
done

