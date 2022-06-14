#include <gol.cuh>
#include <assert.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cmath>

void gameOfLifeTest();
void copyHaloRowsTest();
void copyHaloColumnsTest();
void copyHalos();
bool compareFirstRealAndBottomHaloRow(char* h_life);
bool compareLastRealAndTopHaloRow(char* h_life);

constexpr int generations = 1000;

// The four corners of the grid that contain REAL elements and not halo elements
//constexpr int topLeft     = SIZE + 3;
//constexpr int topRight    = topLeft + SIZE - 1;
//constexpr int bottomLeft  = (SIZE + 2) * SIZE + 1;
//constexpr int bottomRight = bottomLeft + SIZE - 1;

//constexpr int bottomLeftHalo    = (SIZE + 1) * (SIZE + 2) + 1;
//constexpr int bottomRightHalo = bottomLeftHalo + SIZE - 1;

int
main() {
   //copyHaloRowsTest();
   //copyHaloColumnsTest();
   //copyHalos();
   gameOfLifeTest();
   return 0;
}

__host__ void
gameOfLifeTest() {
   char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
   assert(h_life != NULL);
   
   initialState(h_life);
   //printGrid(h_life);
   gameOfLife(&h_life, generations);
}

void
copyHaloRowsTest() {
   char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
   assert(h_life != NULL);
   initialState(h_life);
   //printGrid(SIZE, h_life);

   char* d_life;
   cudaError_t err;
   err = cudaMalloc((void**)&d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char));
   if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate memory on the GPU, error code : %d\n", err);
      fprintf(stderr, cudaGetErrorString(err));
   }
   assert(cudaSuccess == err);

   err = cudaMemcpy(d_life, h_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyHostToDevice);
   assert(cudaSuccess == err);

   dim3 copyBlock = { BLOCK_X, 1, 1 };
   dim3 copyingBlocksRows = calculateCopyingBlocks(SIZE);

   std::cout << "copyBlock " << copyBlock.x << std::endl;
   std::cout << "copyingBlocksRows " << copyingBlocksRows.x << std::endl;

   bool ret = compareLastRealAndTopHaloRow(h_life);
   ret = compareFirstRealAndBottomHaloRow(h_life);

   copyHaloRows<<<copyingBlocksRows, copyBlock>>>(d_life);

   err = cudaDeviceSynchronize();
   assert(cudaSuccess == err);

   err = cudaMemcpy(h_life, d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyDeviceToHost);
   assert(cudaSuccess == err);
   printGrid(h_life);

   cudaFree(d_life);
   free(h_life);
}

void copyHaloColumnsTest() {
   char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
   assert(h_life != NULL);
   initialState(h_life);
   printGrid(h_life);
   std::cout << std:: endl << std::endl;

   char* d_life;
   cudaError_t err;
   err = cudaMalloc((void**)&d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char));
   assert(cudaSuccess == err);

   err = cudaMemcpy(d_life, h_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyHostToDevice);
   assert(cudaSuccess == err);

   dim3 copyBlock = { BLOCK_X, 1, 1 };
   dim3 copyingBlocksColumns = calculateCopyingBlocks(SIZE + 2);
   std::cout << "copyingBlocksColumns " << copyingBlocksColumns.x << std::endl;

   copyHaloColumns<<<copyingBlocksColumns, copyBlock>>>(d_life);

   err = cudaDeviceSynchronize();
   assert(cudaSuccess == err);

   err = cudaMemcpy(h_life, d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyDeviceToHost);
   assert(cudaSuccess == err);

   printGrid(h_life);

   cudaFree(d_life);
   free(h_life);
}

void
copyHalos() {
   char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
   assert(h_life != NULL);
   initialState(h_life);
   printGrid(h_life);

   char* d_life;
   cudaError_t err;
   err = cudaMalloc((void**)&d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char));
   assert(cudaSuccess == err);

   err = cudaMemcpy(d_life, h_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyHostToDevice);
   assert(cudaSuccess == err);

   dim3 copyBlock = { BLOCK_X, 1, 1 };
   dim3 copyingBlocksRows = calculateCopyingBlocks(SIZE);
   dim3 copyingBlocksColumns = calculateCopyingBlocks(SIZE + 2);

   std::cout << "copyingBlocksRows " << copyingBlocksRows.x << std::endl;
   std::cout << "copyingBlocksColumns " << copyingBlocksColumns.x << std::endl;

   copyHaloRows<<<copyingBlocksRows, copyBlock>>>(d_life);
   copyHaloColumns<<<copyingBlocksColumns, copyBlock>>>(d_life);

   err = cudaDeviceSynchronize();
   assert(cudaSuccess == err);

   err = cudaMemcpy(h_life, d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyDeviceToHost);
   assert(cudaSuccess == err);

   bool ret = compareLastRealAndTopHaloRow(h_life);
   assert(ret);

   ret = compareFirstRealAndBottomHaloRow(h_life);
   assert(ret);

   printGrid(h_life);

   cudaFree(d_life);
   free(h_life);
}

bool
compareFirstRealAndBottomHaloRow(char* h_life) {
   // Indices of the first element in the first REAL row, and the
   // first element in the bottom halo row respectively. We do not take
   // into account the corner elements
   int topLeftReal = SIZE + 3;
   int bottomLeftHalo = (SIZE + 1) * (SIZE + 2) + 1;

   for (int i = 0; i < SIZE; i++) {
       if (*(h_life + topLeftReal + i) != *(h_life + bottomLeftHalo + i)) {
          return false;
       }
   }
   return true;
}

bool
compareLastRealAndTopHaloRow(char* h_life) {
   // Indices of the first element in the last REAL row, and the
   // first element in the top halo row respectively. We do not take
   // into account the corner elements
   int bottomLeftReal = SIZE * (SIZE + 2) + 1;
   int topLeftHalo = 1;

   for (int i = 0; i < SIZE; i++) {
       if (*(h_life + bottomLeftReal + i) != *(h_life + topLeftHalo + i)) {
          return false;
       }
   }
   return true;
}
