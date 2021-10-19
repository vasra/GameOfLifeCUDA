#include <gol.cuh>

__global__ void
copyHaloRows(char* d_life) {
   const int threadID = blockIdx.x * blockDim.x + threadIdx.x; // The global ID of the thread, aka its position on the global grid
   
   // threadID must be in the range [1, SIZE]
   if ((threadID >= 1) && (threadID <= SIZE)) {
      d_life[threadID] = d_life[threadID + SIZE * (SIZE + 2)];              // copy last real row to upper halo row
      d_life[threadID + (SIZE + 2) * (SIZE + 1)] = d_life[threadID + SIZE + 2]; // copy first real row to bottom halo row
   }
}

__global__ void
copyHaloColumns(char* d_life) {
   const int threadID = blockIdx.x * blockDim.x + threadIdx.x; // The global ID of the thread, aka its position on the global grid

   if (threadID <= SIZE + 1) {
      d_life[threadID * (SIZE + 2)] = d_life[threadID * (SIZE + 2) + SIZE];       // copy last real column to left halo column
      d_life[threadID * (SIZE + 2) + SIZE + 1] = d_life[threadID * (SIZE + 2) + 1]; // copy first real column to right halo column
   }
}

__global__ void
nextGen(char* d_life, char* d_life_copy) {
   __shared__ char sgrid[BLOCK_Y][BLOCK_X]; // Shared memory grid

   int X = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
   int Y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

   // The global ID of the thread in the grid
   int threadIdGlobal = (SIZE + 2) * Y + X;

   // The local coordinates of the thread in the block
   int x = threadIdx.x;
   int y = threadIdx.y;

   int neighbours = 0;

   if (X <= SIZE + 1 && Y <= SIZE + 1) {
      sgrid[y][x] = d_life[threadIdGlobal];
   }

   __syncthreads();

   // If the thread does not correspond to a halo element inside the block, then calculate its neighbours
   if (x > 0 && x < blockDim.x - 1 && y > 0 && y < blockDim.y - 1) {
      neighbours = sgrid[y - 1][x - 1] + sgrid[y - 1][x]  + sgrid[y - 1][x + 1] +
                sgrid[y][x - 1]    + /* you are here */ sgrid[y][x + 1]    +
                sgrid[y + 1][x - 1] + sgrid[y + 1][x]  + sgrid[y + 1][x + 1];
      
      if ((2 == neighbours && 1 == sgrid[y][x]) || (3 == neighbours)) {
         sgrid[y][x] = 1;
      } else {
         sgrid[y][x] = 0;
      }
      d_life_copy[threadIdGlobal] = sgrid[y][x];
   }
}

__host__ void
printGrid(char* h_life) {
  for (int i = 0; i < SIZE + 2; i++) {
   for (int j = 0; j < SIZE + 2; j++) {
     printf("%d", *(h_life + i * (SIZE + 2) + j));
     if (j == SIZE + 1) {
      printf("\n");
     }
   }
  }
}

/////////////////////////////////////////////////////////////////
// Randomly produces the first generation. The living organisms
// are represented by a 1, and the dead organisms by a 0.
/////////////////////////////////////////////////////////////////
__host__ void
initialState(char* h_life) {
   float randomProbability = 0.0f;
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> probability(0.0f, 1.0f);

   for (int i = 0; i < SIZE + 2; i++) {
      for (int j = 0; j < SIZE + 2; j++) {
         // Initialize all halo values to 0. The rest will be assigned values randomly.
         if (0 == i || SIZE + 1 == i || 0 == j || SIZE + 1 == j) {
            *(h_life + i * (SIZE + 2) + j) = 0;
         } else {
            randomProbability = static_cast<float>(probability(gen));
            if (randomProbability >= 0.5f) {
               *(h_life + i * (SIZE + 2) + j) = 1;
            } else {
               *(h_life + i * (SIZE + 2) + j) = 0;
            }
         }
      }
   }
}

__host__ dim3
calculateCopyingBlocks(int size) {
   float gridSize = static_cast<float>(size);
   float fBlockDim = static_cast<float>(BLOCK_X);
   unsigned int x = static_cast<unsigned int>(ceil(gridSize / fBlockDim));
   return { x, 1, 1 };
}

//////////////////////////////////////////////////////////////////////////////////////
// Plays the Game Of Life. It checks the contents of d_life,
// calculates the results, and stores them in d_life_copy. The living organisms
// are represented by a 1, and the dead organisms by a 0.
//////////////////////////////////////////////////////////////////////////////////////
__host__ void
gameOfLife(char* h_life, int generations, float* msecs) {
   // The grids that will be copied to the GPU
   char* d_life;
   char* d_life_copy;
   cudaError_t err;

   cudaFuncSetCacheConfig(nextGen, cudaFuncCachePreferShared);

   err = cudaMalloc((void**)&d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return;
   }

   err = cudaMemcpy(d_life, h_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyHostToDevice);
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
      return;
   }

   err = cudaMalloc((void**)&d_life_copy, (SIZE + 2) * (SIZE + 2) * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return;
   }

   // How many blocks will be used to copy the halo rows and columns respectively
   dim3 copyBlock = { BLOCK_X, 1, 1 };
   dim3 copyingBlocksRows = calculateCopyingBlocks(SIZE);
   dim3 copyingBlocksColumns = calculateCopyingBlocks(SIZE + 2);

   // The layout of the threads in the block. Depends on
   // how many threads we decide to use for each block
   dim3 threadsInBlock{ BLOCK_X, BLOCK_Y, 1 };

   // The layout of the blocks in the grid. We subtract 2 from each
   // coordinate, to compensate for the halo rows and columns of each block
   unsigned int gridX = static_cast<int>(ceil(SIZE / static_cast<float>(threadsInBlock.x - 2)));
   unsigned int gridY = static_cast<int>(ceil(SIZE / static_cast<float>(threadsInBlock.y - 2)));
   dim3 gridDims{ gridX, gridY, 1 };

   timestamp t_start = getTimestamp();

   for (int gen = 0; gen < generations; gen++) {
      copyHaloRows <<<copyingBlocksRows, copyBlock>>> (d_life);
      copyHaloColumns <<<copyingBlocksColumns, copyBlock>>> (d_life);
      nextGen <<<gridDims, BLOCK_X * BLOCK_Y>>> (d_life, d_life_copy);

      /////////////////////////////////////////////////////////////////////////////////////////////////
      // Swap the addresses of the two tables. That way, we avoid copying the contents
      // of d_life to d_life_copy. Each round, the addresses are exchanged, saving time from running
      // a loop to copy the contents.
      /////////////////////////////////////////////////////////////////////////////////////////////////
      std::swap(d_life, d_life_copy);
   }

   err = cudaDeviceSynchronize();
   if (cudaSuccess != err) {
      fprintf(stderr, "Error synchronizing devices: %d\n", err);
      return;
   }

   *msecs = getElapsedtime(t_start);

   err = cudaMemcpy(h_life, d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyDeviceToHost);
   printGrid(d_life);
   printf("\n");
   printGrid(d_life_copy);

   err = cudaFree(d_life);
   if (cudaSuccess != err) {
      fprintf(stderr, "Error freeing GPU memory: %d\n", err);
      return;
   }

   err = cudaFree(d_life_copy);
   if (cudaSuccess != err) {
      fprintf(stderr, "Error freeing GPU memory: %d\n", err);
      return;
   }
}
