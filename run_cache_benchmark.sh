#!/bin/bash

# DESCRIPTION :
# Runs Cachegrind in a localized way to better extract cache behaviour during the execution of the selected transpose() method

# OUTPUT :
# prompts on terminal :
# - Total Memory reads
# - L1 cache misses on reads
# - L2 cache misses on reads
# - Total Memory writes
# - L1 cache misses on write
# - L2 cache misses on write
# - % of read misses
# - % of write misses


### User variables ###
N=12                # defines the matrix size : ( 2^N x 2^N )

method="blocks"     # chose which version of transpose() to run between "transpose_naive()" and "transpose_blocks()" by setting :
                    #   method="naive" : run transpose_naive()
                    #   method="blocks" : run transpose_blocks()

block_size=4       # in method transpose_blocks() defines the block size : ( 2^block_size x 2^block_size )





#### Gather Valgrind data ####

# Run Valgrind with cachegrind tool
EXEC='./bin/homework-1 --method='$method' --N='$N' --block_size='$block_size
FUNCTION="transpose_naive|transpose_blocks"

echo "running Cachegrind on '$EXEC'..."
valgrind --tool=cachegrind $EXEC &> /dev/null

# Annotate the specified functions and extract relevant data
cg_annotate cachegrind.out.* | grep -E $FUNCTION > function_stats.txt

# Loop through each line in function_stats.txt
while IFS= read -r line; do
    # Split the line into columns based on spaces
    columns=($line)
    # Assign values to variables
    Dr=$(echo "${columns[3]}" | tr -d ',')
    D1mr=$(echo "${columns[4]}" | tr -d ',')
    D2mr=$(echo "${columns[5]}" | tr -d ',')
    Dw=$(echo "${columns[6]}" | tr -d ',')
    D1mw=$(echo "${columns[7]}" | tr -d ',')
    D2mw=$(echo "${columns[8]}" | tr -d ',')
    method=$(echo "${columns[9]}" | tr -d ',')

    echo
    echo "Method : $method"
    
    # Print Values
    echo "  Total Reads: $Dr"
    echo "  L1 Read Misses: $D1mr"
    echo "  L2 Read Misses: $D2mr"
    echo "  Total Writes: $Dw"
    echo "  L1 Write Misses: $D1mw"
    echo "  L2 Write Misses: $D2mw"
    echo

    # Compute average % read misses and % write misses
    total_read_misses=$((D1mr + D2mr))
    total_write_misses=$((D1mw + D2mw))
    total_memory_accesses=$((Dr + Dw))
    echo "  Total Read Misses: $total_read_misses"
    echo "  Total Write Misses: $total_write_misses"
    echo "  Total Memory Accesses: $total_memory_accesses"
    echo

    avg_read_misses=$(echo "scale=3; $total_read_misses / $Dr * 100" | bc)
    avg_write_misses=$(echo "scale=3; $total_write_misses / $Dw  * 100" | bc)
    echo "  Average % Read Misses: $avg_read_misses%"
    echo "  Average % Write Misses: $avg_write_misses%"

done < function_stats.txt

# Clean up
rm cachegrind.out.* function_stats.txt
