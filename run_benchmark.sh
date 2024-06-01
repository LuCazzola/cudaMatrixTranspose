#!/bin/bash

# DESCRIPTION :
# "benchmark" runs one or more matrix transposition algorithms : 
# - The program runs transpose_naive() or transpose_blocks() (or both) depending on the "method" parameter
# - each algorithm is performed with different parameters configuration
# - each configuration runs "iterations_per_config", then the matrix size is increased by one : 
#      - starting with (2^min_powerof2 x 2^min_powerof2) up to (2^max_powerof2 x 2^max_powerof2) 
# - during each configuration some data are stored into .csv files which can be found into the "data" folder

# OUTPUT :
# one or more .csv files containing :
#  - Details about the run such as : block_size used (only for transpose_blocks()), optimization flags used, data type of matrix elements
#  - a table which stores : 
#      - { execution time, effective bandwidth } "iterations_per_config" times per configuration
#      - { average execution time, average effective bandwidth } (per each configuration)
#      - { standard deviation of execution time, standard deviation of effective bandwidth } (per each configuration)  


### User Variables ###
min_powerof2=6           # size of the FIRST tested matrix : ( 2^min_powerof2 x 2^min_powerof2 ) 
max_powerof2=12          # size of the LAST tested matrix : ( 2^max_powerof2 x 2^max_powerof2 )
iterations_per_config=10 # number of times each configuration of parameters is repeated executed

method="all"     # chose which version of transpose to run between "transpose_naive()" and "transpose_blocks()" by setting :
                 #   method="naive" : run transpose_naive() generating 1 .csv
                 #   method="blocks" : run transpose_blocks() generating 1 .csv
                 #   method="all" : runs both transpose_naive() and transpose_blocks() generating 2 .csv

block_size=4             # in method transpose_blocks() defines the block size : ( 2^block_size x 2^block_size )

./bin/benchmark  --method=$method --min_powerof2=$min_powerof2 --max_powerof2=$max_powerof2 --iterations_per_config=$iterations_per_config --block_size=$block_size
