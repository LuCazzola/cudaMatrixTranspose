# GPU computing : Homework - 2

The following repository contains all the material related to the **SECOND homework** related to **Matrix Transposition** assigned during the GPU computing course : University of Trento (Italy) a.y. 2023/2024.
<br>
To take a look to the first homewor, please check the [**repo**](https://github.com/LuCazzola/GPUcomputing-Homework1)
<br>

To see the report and better understand what this work is about, click [**Here**](materials/Luca-Cazzola_HW-1.pdf)

![Matrix Transposition](materials/problem-intro.png)

<br>

## How to use

Download the directory
```
git clone https://github.com/LuCazzola/GPUcomputing-Homework2.git
cd GPUcomputing-Homework2
```
<br>


Here follows the Hierarchy of relevant project's files :
```bash

.
├── bin                         # final executables
│    └── ...
├── obj                         # intermediate object files
│    └── ...
└── src                         # source code
│    ├── headers                # header files
│    │    └── ...                         
│    ├── benchmark.c            # produce an output file according to options in "run_benchmark.sh"
│    ├── main.c                 # test the functions according to options in "run_main.sh"
│    ├── matrix.c               # definition of methods to handle matrices
│    ├── transpose.c            # functions to compute the transpose of a given matrix
│    └── utils.c                # common functions and parameter parsing
├── data                        # data gathered via "run_benchmark.sh"
│    └── ...
├── run_benchmark.sh            # set parameters related to "benchmark.c" and run the script
├── run_main.sh                 # set parameters related to "main.c" and run the script
├── run_cache_benchmark.sh      # run cachegrind to benchmark cache miss % on specified function
├── Makefile
└── ...
```
<br>
