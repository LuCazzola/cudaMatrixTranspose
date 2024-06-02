# Matrix transposition : from sequential to parallel with CUDA

The following repository contains all the material related to both the homeworks on **Matrix Transposition** assigned during the GPU computing course : University of Trento (Italy) a.y. 2023/2024.
<br>

To see the report and better understand what this work is about, click [**Here**](materials/LC-GPU_computing-report.pdf)

![Matrix Transposition](materials/problem-intro.png)

<br>

## How to use

Download the directory
```
git clone https://github.com/LuCazzola/cudaMatrixTranspose.git
cd cudaMatrixTranspose
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
│    ├── benchmark_gpu.cu       # produce an output file according to options in "launch_benchmark.sh"
│    .
│    ├── main.c                 # test the functions according to options in "run_main.sh"
│    ├── main_gpu.cu            # test the functions according to options in "launch_main.sh"
│    .
│    ├── transpose.c            # functions to compute the transpose of a given matrix
│    ├── transpose_gpu.c 
│    .
│    ├── matrix.c               # definition of methods to handle matrices
│    ├── opt_parser.c           # command line parameter parsing
│    .
│    └── common_cuda.cu         # defines some common functions for cuda methods
│
├── run_benchmark.sh            # set parameters related to "benchmark.c" and run the script
├── run_main.sh                 # set parameters related to "main.c" and run the script
├── run_cache_benchmark.sh      # run cachegrind to benchmark cache miss % on specified function
│   .
├── launch_benchmark.sh         # set parameters related to "benchmark_gpu.cu" and run the script on SLURM system
├── launch_main.sh              # set parameters related to "main_gpu.cu" and run the script on SLURM system
│   .
├── data                        # data gathered via "run_benchmark.sh" & "launch_benchmark.sh"
│    └── ...
├── plot_data.py                # generates graphs using the data stored in "data" folder
│
├── Makefile
└── ...
```
<br>

### Main commands

Makefile defines 4 rules :
* **make** : builds object files and **homework-1 + homework-2** executables
* **make debug** :  builds object files and ALL executables adding debugging flags
* **make benchmark** : builds object files and **benchmark + benchmark_gpu** executable
* **make clean** : cleans all object files
<br>
There are many pre-set scripts to choose from :
<br>
>> <a href="#CPU-sec">CPU scripts section ( Homework-1 )</a>
<br>
>> <a href="#GPU-sec">GPU scripts section ( Homework-2 )</a> 

<hr><br>

<a name="CPU-sec"></a>
## CPU test commands ( Homework-1 )
### COMMANDS
**"run_main.sh"** script sets **parameters** related to **homework-1** executable and runs it.
<br>
To [change run parameters](run_main.sh?plain=1#L12-L19) and have a better understanding of its functionalities see : [**run_main.sh**](run_main.sh?plain=1#L3-L9)
```
make
./run_main.sh
```

<br>

**"run_benchmark.sh"** script sets **parameters** related to **benchmark** executable and runs it.
<br>
extracted data can be found on the [**data folder**](data/)
<br>
To [change run parameters](run_benchmark.sh?plain=1#L20-L28) and have a better understanding of its functionalities see : [**run_benchmark.sh**](run_benchmark.sh?plain=1#L3-L17)
```
make benchmark
./run_benchmark.sh
```

<br>

**"run_cache_benchmark.sh"** script sets **parameters** related to **homework-1** and runs Cachegrind on it, extracting localized informations about cache misses inside transpose_naive() or transpose_blocks() functions (according to the chosen parameter "method")
<br>
To [change run parameters](run_cache_benchmark.sh?plain=1#L18-L25) and have a better understanding of its functionalities see : [**run_cache_benchmark.sh**](run_cache_benchmark.sh?plain=1#L3-L15)
```
make clean
make debug
./run_cache_benchmark.sh
```
<hr><br>

<a name="GPU-sec"></a>
## GPU test commands ( Homework-2 )
### NOTE
Please consider that the following commands are supposed to be ran on the **Marzola DISI cluster**, modify the [launch_main.sh](launch_main.sh?plain=1#L3-L13) & [launch_benchmark.sh](launch_benchmark.sh?plain=1#L3-L13) scripts if needed to change partition or SLURM system.
<br><br>
**Outside the cloned project folder** upload the project's directory to the login node
```
scp -r cudaMatrixTranspose <YOUR USERNAME>@marzola.disi.unitn.it:/home/<YOUR USERNAME>
```
Then login and go inside the project's folder
```
cd cudaMatrixTranspose
module load cuda
```
<hr><br>

### COMMANDS
**"launch_main.sh"** script sets **parameters** related to **homework-2** executable and runs it.
<br>
To [change run parameters](launch_main.sh?plain=1#L26-L37) and have a better understanding of its functionalities see : [**launch_main.sh**](launch_main.sh?plain=1#L17-L23)
```
make
sbatch launch_main.sh
```
To visualize the results, once the node returns do:
```
cat output.out
```
<br>

**"launch_benchmark.sh"** script sets **parameters** related to **benchmark_gpu** executable and runs it.
<br>
extracted data can be found on the [**data folder**](data/)
<br>
To [change run parameters](launch_benchmark.sh?plain=1#L35-L49) and have a better understanding of its functionalities see : [**launch_benchmark.sh**](launch_benchmark.sh?plain=1#L16-L30)
```
make benchmark
sbatch launch_benchmark.sh
```

<hr><br>

## Graph Plotting
Inside the project's directory there's also a python script which take's the content of [**data folder**](data/) and generates 2 types of graphs

* x : Matrix size - y : Mean execution time
* x : Matrix size - y : Mean effective bandwidth
  
<br>

Test it by :
```
python3 plot_data.py
```
You can customize what information to plot inside the [**script**](plot_data.py?plain=1#L53-L74)


<hr><br>

## Extra Customization
It's also possible to change some other parameters at compilation level (optimization level and matrix element data type) by changing some [**variables in the makefile**](Makefile?plain=1#L3-L6)) :


