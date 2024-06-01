#!/bin/bash

### Options to run the test on the Marzola cluster ###

#SBATCH --job-name=benchmark_gpu-run
#SBATCH --output=output.out
#SBATCH --error=error.err

#SBATCH --partition=edu5
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


# DESCRIPTION :
# "benchmark_gpu" runs matrix transposition on GPU algorithm : 
# - The program runs transpose_blocks_gpu() / transpose_blocks_gpu_coalesced() kernels
# - The kernels are performed with different parameters configuration
# - each configuration runs "iterations_per_config", then the matrix size is increased by one : 
#      - starting with (2^min_powerof2 x 2^min_powerof2) up to (2^max_powerof2 x 2^max_powerof2) 
# - during each configuration some data are stored into .csv files which can be found into the "data" folder

# OUTPUT :
# one or more .csv files containing :
#  - Details about the run such as : block_size, th_size_x, th_size_y, & data type of matrix elements
#  - a table which stores : 
#      - { execution time, effective bandwidth } "iterations_per_config" times per configuration
#      - { average execution time, average effective bandwidth } (per each configuration)
#      - { standard deviation of execution time, standard deviation of effective bandwidth } (per each configuration)  


### User Variables ###

method="all"    # chose which version of transpose to run between "transpose_blocks_gpu()" and "transpose_blocks_gpu_coalesced()" by setting :
                #   method="blocks_naive" : run transpose_blocks_gpu() generating 1 .csv
                #   method="blocks_coalesced" : run transpose_blocks_gpu_coalesced() generating 1 .csv
                #   method="all" : runs both transpose_blocks_gpu() and transpose_blocks_gpu_coalesced() generating 2 .csv


min_powerof2=6            # size of the FIRST tested matrix : ( 2^min_powerof2 x 2^min_powerof2 ) 
max_powerof2=12           # size of the LAST tested matrix : ( 2^max_powerof2 x 2^max_powerof2 )
iterations_per_config=500 # number of times each configuration of parameters is repeated executed


block_size=5  # in method transpose_blocks() defines the block size : ( 2^block_size x 2^block_size )

th_size_x=3  # defines threadBlock shape ( 2^th_size_x x 2^th_size_y )
th_size_y=5

./bin/benchmark_gpu  --method=$method --min_powerof2=$min_powerof2 --max_powerof2=$max_powerof2 --iterations_per_config=$iterations_per_config --block_size=$block_size --th_size_x=$th_size_x --th_size_y=$th_size_y 
