#ifndef _GOL_CUH_
#define _GOL_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <random>
#include <lcutil.h>
#include <timestamp.h>
#include <device_launch_parameters.h>

__host__ void
printGrid(int size, char* h_life);

__host__ void
initialState(int size, char* h_life);

__global__ void
copyHaloRows(char* d_life, const int size);

__global__ void
copyHaloColumns(char* d_life, const int size);

__global__ void
nextGen(char* d_life, char* d_life_copy, int size);

#endif
