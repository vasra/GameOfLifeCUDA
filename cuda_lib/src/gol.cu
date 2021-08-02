#include <gol.cuh>

__global__ void
copyHaloRows(char* d_life, const int size) {
    // The global ID of the thread, aka its position on the 2D grid
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // The index of the first and last elements of the bottom halo respectively.
    // We do not take into account the corner elements
    const int bottomLeftHalo = (size + 1) * (size + 2) + 1;
    const int bottomRightHalo = bottomLeftHalo + size - 1;
    
    // threadID must be in the range [1, size]
    if (threadID >= 1 && threadID <= size) {
        d_life[threadID] = d_life[threadID + size * (size + 2)]; // copy bottom row to upper halo row
    } else if (threadID >= bottomLeftHalo && threadID <= bottomRightHalo) {
        d_life[threadID] = d_life[threadID % (size + 2)];  // copy upper row to bottom halo row
    }
}

__global__ void
copyHaloColumns(char* d_life, const int size) {
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    // threadID must either be a multiple of size + 2 (left column),
    // or 
    if (threadID % (size + 2) == 0) {
        d_life[threadID] = d_life[threadID + size + 1]; // copy rightmost column to left halo column
    } else if ((threadID + 1) % (size + 2) == 0) {
        d_life[threadID] = d_life[threadID - size];     // copy leftmost column to right halo column
    }

    // copy corner elements
    // 1. bottom right -> top left
    // 2. bottom left  -> top right
    // 3. top left     -> bottom right
    // 4. top right    -> bottom left
//     if (0 == threadID) {
//         d_life[threadID] = d_life[(size + 2) * size + size];
//     } else if (size + 1 == threadID) {
//         d_life[threadID] = d_life[(size + 2) * size + 1];
//     } else if ((size + 2) * (size + 1) + size + 1 == threadID) {
//         d_life[threadID] = d_life[size + 3];
//     } else if ((size + 2) * size + 1 == threadID) {
//         d_life[threadID] = d_life[size];
//     }
}

__global__ void
nextGen(char* d_life, char* d_life_copy, int size) {
    // Shared memory grid
    extern __shared__ char sgrid[];

    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    // The global ID of the thread in the grid
    int threadIdGlobal = (size + 2) * Y + X;

    // The local ID of the thread in the block
    int threadIdLocal = threadIdx.y * blockDim.x + threadIdx.x;

    int neighbours;

    if (X <= size + 1 && Y <= size + 1) {
        sgrid[threadIdLocal] = d_life[threadIdGlobal];
    }

    __syncthreads();

    // If the thread does not correspond to a halo element, then calculate its neighbours
    if (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1) {
        neighbours = sgrid[threadIdLocal - blockDim.x - 1] + sgrid[threadIdLocal - blockDim.x] + sgrid[threadIdLocal - blockDim.x + 1] +
                     sgrid[threadIdLocal - 1]              + /* you are here */                  sgrid[threadIdLocal + 1]              +
                     sgrid[threadIdLocal + blockDim.x - 1] + sgrid[threadIdLocal + blockDim.x] + sgrid[threadIdLocal + blockDim.x + 1];
    }

    if ((2 == neighbours && 1 == sgrid[threadIdLocal]) || (3 == neighbours)) {
        sgrid[threadIdLocal] = 1;
    } else {
        sgrid[threadIdLocal] = 0;
    }
}

__host__ void
printGrid(int size, char* h_life) {
  for (int i = 0; i < size + 2; i++) {
    for (int j = 0; j < size + 2; j++) {
      printf("%d", *(h_life + i * (size + 2) + j));
      if (j == size + 1) {
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
initialState(int size, char* h_life) {
    float randomProbability = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> probability(0.0f, 1.0f);

    for (int i = 0; i < size + 2; i++) {
        for (int j = 0; j < size + 2; j++) {
            // Initialize all halo values to 0. The rest will be assigned values randomly.
            if (0 == i || size + 1 == i || 0 == j || size + 1 == j) {
                *(h_life + i * (size + 2) + j) = 0;
            } else {
                randomProbability = static_cast<float>(probability(gen));
                if (randomProbability >= 0.5f) {
                    *(h_life + i * (size + 2) + j) = 1;
                } else {
                    *(h_life + i * (size + 2) + j) = 0;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// Plays the Game Of Life. It checks the contents of d_life,
// calculates the results, and stores them in d_life_copy. The living organisms
// are represented by a 1, and the dead organisms by a 0.
//////////////////////////////////////////////////////////////////////////////////////
__host__ float
gameOfLife(const int size, char* h_life, int generations, int threads) {
    // The grids that will be copied to the GPU
    char* d_life;
    char* d_life_copy;
    cudaError_t err;

    err = cudaMalloc((void**)&d_life, (size + 2) * (size + 2) * sizeof(char));
    if (cudaSuccess != err) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return static_cast<float>(err);
    }

    err = cudaMemcpy(d_life, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaSuccess != err) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return static_cast<float>(err);
    }

    err = cudaMalloc((void**)&d_life_copy, (size + 2) * (size + 2) * sizeof(char));
    if (cudaSuccess != err) {
        fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
        return static_cast<float>(err);
    }

    err = cudaMemcpy(d_life_copy, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaSuccess != err) {
        fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
        return static_cast<float>(err);
    }

    // How many blocks will be used to copy the halo rows and columns respectively
    int copyingBlocksRows = size / threads;
    int copyingBlocksColumns = ceil((size + 2) / threads);

    // The layout of the threads in the block. Depends on
    // how many threads we decide to use for each block
    dim3 threadsInBlock;
    switch (threads) {
    case 16:
        threadsInBlock.x = 4;
        threadsInBlock.y = 4;
        threadsInBlock.z = 1;
        break;
    case 32:
        threadsInBlock.x = 8;
        threadsInBlock.y = 4;
        threadsInBlock.z = 1;
        break;
    case 64:
        threadsInBlock.x = 8;
        threadsInBlock.y = 8;
        threadsInBlock.z = 1;
        break;
    case 128:
        threadsInBlock.x = 16;
        threadsInBlock.y = 8;
        threadsInBlock.z = 1;
        break;
    case 256:
        threadsInBlock.x = 16;
        threadsInBlock.y = 16;
        threadsInBlock.z = 1;
        break;
    case 512:
        threadsInBlock.x = 32;
        threadsInBlock.y = 16;
        threadsInBlock.z = 1;
        break;
    default:
        break;
    }

    // The layout of the blocks in the grid. We subtract 2 from each
    // coordinate, to compensate for the halo rows and columns
    unsigned int gridX = static_cast<int>(ceil(size / (threadsInBlock.x - 2)));
    unsigned int gridY = static_cast<int>(ceil(size / (threadsInBlock.y - 2)));
    dim3 gridDims{ gridX, gridY, 1 };

    // The number of bytes of shared memory that the block will use
    unsigned int sharedMemBytes = threadsInBlock.x * threadsInBlock.y * sizeof(char);

    timestamp t_start = getTimestamp();

    for (int gen = 0; gen < generations; gen++) {
        copyHaloRows <<<copyingBlocksRows, threads>>> (d_life, size);
        copyHaloColumns <<<copyingBlocksColumns, threads>>> (d_life, size);
        nextGen <<<gridDims, threads, sharedMemBytes>>> (d_life, d_life_copy, size);

        /////////////////////////////////////////////////////////////////////////////////////////////////
         // Swap the addresses of the two tables. That way, we avoid copying the contents
         // of d_life to d_life_copy. Each round, the addresses are exchanged, saving time from running
         // a loop to copy the contents.
         /////////////////////////////////////////////////////////////////////////////////////////////////
        std::swap(d_life, d_life_copy);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error synchronizing devices: %d\n", err);
        return static_cast<float>(err);
    }

    float msecs = getElapsedtime(t_start);

    err = cudaFree(d_life);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %d\n", err);
        return static_cast<float>(err);
    }

    err = cudaFree(d_life_copy);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing GPU memory: %d\n", err);
        return static_cast<float>(err);
    }

    return msecs;
}
