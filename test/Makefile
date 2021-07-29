CUDA_INSTALL_PATH = /usr/local/cuda-11.4
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
INCDIR = -I../cuda_lib/cuda_lib/include
LIBS = -L/usr/local/cuda-11.4/lib64/ -lcudart -lcuda
FLAGS = $(OPTFLAG) -I$(CUDA_INSTALL_PATH)/include -Wall -g $(INCDIR)
NVFLAGS = -O2 -I$(CUDA_INSTALL_PATH)/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g $(INCDIR)
BITS = $(shell getconf LONG_BIT)
ifeq ($(BITS),64)
        LIBSUFFIX := 64
endif
LFLAGS = -L$(CUDA_INSTALL_PATH)/lib$(LIBSUFFIX) -lm -lstdc++ -lcudart
CLEAN_FILES = gol.a

# Output directory
OUTPUT_DIR = bin

# CUDA library
CUDA_LIBRARY = -L../cuda_lib/cuda_lib/bin/gol.a

EXEC ?= @echo "[@]"

all: lifeTest.o lifeTest

lifeTest.o: lifeTest.cu
	$(EXEC) $(NVCC) $(NVFLAGS) $(CUDA_LIBRARY) $(LIBS) -DUNIX -c $^ -o $@
	mkdir -p $(OUTPUT_DIR)
	mv $@ $(OUTPUT_DIR)/$@

lifeTest: $(OUTPUT_DIR)/lifeTest.o
		$(NVCC) $(NVFLAGS) $(INCDIR) $(LFLAGS) $(CUDA_LIBRARY) $(LIBS) $^ -o $@
		mv $@ $(OUTPUT_DIR)/$@

clean:
	\rm -f bin/*

#rebuild: clean lifeTest
