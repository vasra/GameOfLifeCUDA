CUDA_INSTALL_PATH = /usr/local/cuda-11.4
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -Iinclude
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
        LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart
CLEAN_FILES = gol.a

# Output directory
OUTPUT_DIR = bin

# CUDA library
CUDA_LIBRARY = gol.a

# CUDA source path
CUDA_SRC = src

$(CUDA_LIBRARY): ${CUDA_SRC}/gol.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@
	mkdir -p $(OUTPUT_DIR)
	mv $(CUDA_LIBRARY) $(OUTPUT_DIR)/$(CUDA_LIBRARY)

clean:
	\rm -f bin/*

#rebuild: clean lifeTest
