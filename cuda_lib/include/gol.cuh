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

constexpr int SIZE = 840;   // the size of one side of the square grid
constexpr int BLOCK_X = 32; // the size of the x dimension of a block
constexpr int BLOCK_Y = 16; // the size of the y dimension of a block

__global__ void
copyHaloRows(char* d_life);

__global__ void
copyHaloColumns(char* d_life);

__global__ void
nextGen(char* d_life, char* d_life_copy);

__host__ void
printGrid(char* h_life);

__host__ void
initialState(char* h_life);

__host__ dim3
calculateCopyingBlocks(int size);

__host__ void
gameOfLife(char** h_life, int generations);

#endif
