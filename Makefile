#### User defined Variables ####

# choose among {-O0 -O1 -O2 -O3}
ADDITIONAL_OPTIM_FALG := -O2
# chose among {"int", "float"}
MATRIX_ELEM_DTYPE := float

#### General Variables ####
SOURCEDIR := src
BUILDDIR := obj
TARGETDIR := bin

CC := gcc
NVCC := nvcc

INCLUDE := -I$(SOURCEDIR)/headers

OPT := -std=c11 -Wall -lm 
NVCC_FLAGS := -Xcompiler -Wall -lm -lcurand

# Main executable to test function
BUILDNAME := homework-1
# Main executable to run GPU
BUILDGPUNAME := homework-2
# File to build CPU statistics for analysis purposes
BUILDBENCH := benchmark
# File to build GPU statistics for analysis purposes
BUILDGPUBENCH := benchmark_gpu

MAIN := main.c
GPUMAIN := main_gpu.cu

BENCHMARK := benchmark.c
GPUBENCHMARK := benchmark_gpu.cu

OBJECTS := \
	$(BUILDDIR)/matrix.o \
	$(BUILDDIR)/transpose.o \
	$(BUILDDIR)/opt_parser.o

GPU_OBJECTS := \
	$(BUILDDIR)/common_cuda.o \
	$(BUILDDIR)/transpose_gpu.o

##
#### Rules ####
##

# All rule builds Both Homework-1 and Homework-2
all: OPT += $(ADDITIONAL_OPTIM_FALG)
all: $(TARGETDIR)/$(BUILDNAME)
all: $(TARGETDIR)/$(BUILDGPUNAME)

benchmark: $(TARGETDIR)/$(BUILDBENCH)
benchmark: $(TARGETDIR)/$(BUILDGPUBENCH)

debug: OPT += -DDEBUG -g
debug: NVCC_FLAGS += -DDEBUG -G 
debug: all
debug: benchmark

##
### GCC builds ###
##

# build intermediate object files
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c Makefile
	@mkdir -p $(BUILDDIR) $(TARGETDIR)
	@$(CC) -c -o $@ $(INCLUDE) $< $(OPT) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)'
	@echo building: $<

# build main with references
$(TARGETDIR)/$(BUILDNAME): $(SOURCEDIR)/$(MAIN) $(OBJECTS)
	@mkdir -p $(@D)
	@$(CC) $^ -o $@ $(INCLUDE) $(OPT) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)'
	@echo building main into: $@

# build benchmark file
$(TARGETDIR)/$(BUILDBENCH): $(SOURCEDIR)/$(BENCHMARK) $(OBJECTS)
	@mkdir -p $(@D)
	@$(CC) $^ -o $@ $(INCLUDE) $(OPT) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)'
	@echo building benchmark into: $@


##
### NVCC builds ###
##

# build intermediate object files
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cu Makefile
	@mkdir -p $(BUILDDIR) $(TARGETDIR)
	@$(NVCC) -c -o $@ $(INCLUDE) $(NVCC_FLAGS) $< -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)'
	@echo building: $<

# build GPU main with references
$(TARGETDIR)/$(BUILDGPUNAME): $(SOURCEDIR)/$(GPUMAIN) $(OBJECTS) $(GPU_OBJECTS)
	@mkdir -p $(@D)
	@$(NVCC) $^ -o $@ $(INCLUDE) $(NVCC_FLAGS) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)'
	@echo building GPU main into: $@

# build GPU benchmark with references
$(TARGETDIR)/$(BUILDGPUBENCH): $(SOURCEDIR)/$(GPUBENCHMARK) $(OBJECTS) $(GPU_OBJECTS)
	@mkdir -p $(@D)
	@$(NVCC) $^ -o $@ $(INCLUDE) $(NVCC_FLAGS) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)'
	@echo building GPU benchmark into: $@


clean:
	@rm $(BUILDDIR)/*.o $(TARGETDIR)/*
	@rm -r $(TARGETDIR)
	@echo directories cleaned

.PHONY: all debug clean gpu_build
