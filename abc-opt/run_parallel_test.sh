#!/bin/bash

# Create the output directory
mkdir ../output

echo "=========================================" >> output_file.txt
echo "=========================================" >> output_file.txt
date >> output_file.txt
echo "=========================================" >> output_file.txt

# Run the time_numparticles test for all thread number
for threads in 1 2 4 6 8 10 12 14 16 18 20
do
	export OMP_NUM_THREADS=$threads

	./test time_numparticles >> output_file.txt 2>&1
done

# Run the optimize (always serial) to produce both history and simulation data
./test optimize >> output_file.txt 2>&1
