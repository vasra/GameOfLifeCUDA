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

// the SIZE of the grid without the halos
constexpr int generations = 1;

// The four corners of the grid that contain REAL elements and not halo elements
constexpr int topLeft     = SIZE + 3;
constexpr int topRight    = topLeft + SIZE - 1;
constexpr int bottomLeft  = (SIZE + 2) * SIZE + 1;
constexpr int bottomRight = bottomLeft + SIZE - 1;

constexpr int bottomLeftHalo  = (SIZE + 1) * (SIZE + 2) + 1;
constexpr int bottomRightHalo = bottomLeftHalo + SIZE - 1;

int
main() {
    std::cout << "Hello world test!" << std::endl;
    //copyHaloRowsTest();
    //copyHaloColumnsTest();
    //copyHalos();
    //gameOfLifeTest();
    return 0;
}

__host__ void
gameOfLifeTest() {
  char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
  assert(h_life != NULL);
  initialState(SIZE, h_life);
  printGrid(SIZE, h_life);

  gameOfLife(SIZE, h_life, generations, THREADS);  
}

void
copyHaloRowsTest() {
  char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
  assert(h_life != NULL);
  initialState(SIZE, h_life);
  //printGrid(SIZE, h_life);

  char* d_life;
  cudaError_t err;
  err = cudaMalloc((void**)&d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char));
  assert(cudaSuccess == err);

  err = cudaMemcpy(d_life, h_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyHostToDevice);
  assert(cudaSuccess == err);

  int copyingBlocksRows = SIZE / THREADS;
  std::cout << "copyingBlocksRows " << copyingBlocksRows << std::endl;
  bool ret = compareLastRealAndTopHaloRow(h_life, SIZE);
  ret = compareFirstRealAndBottomHaloRow(h_life, SIZE);

  copyHaloRows<<<copyingBlocksRows, THREADS>>>(d_life, SIZE);

  err = cudaDeviceSynchronize();
  assert(cudaSuccess == err);

  err = cudaMemcpy(h_life, d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == err);
  printGrid(SIZE, h_life);

  cudaFree(d_life);
  free(h_life);
}

void copyHaloColumnsTest() {
    char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
    assert(h_life != NULL);
    initialState(SIZE, h_life);
    printGrid(SIZE, h_life);
    std::cout << std:: endl << std::endl;

    char* d_life;
    cudaError_t err;
    err = cudaMalloc((void**)&d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char));
    assert(cudaSuccess == err);

    err = cudaMemcpy(d_life, h_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyHostToDevice);
    assert(cudaSuccess == err);

    int copyingBlocksColumns = SIZE + 2) * (SIZE + 2) / THREADS;
    std::cout << "copyingBlocksColumns " << copyingBlocksColumns << std::endl;

    copyHaloColumns<<<copyingBlocksColumns, THREADS>>>(d_life, SIZE);

    err = cudaDeviceSynchronize();
    assert(cudaSuccess == err);

    err = cudaMemcpy(h_life, d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == err);

    printGrid(SIZE, h_life);

    cudaFree(d_life);
    free(h_life);
}

void
copyHalos() {
  char* h_life = (char*)malloc((SIZE + 2) * (SIZE + 2) * sizeof(char));
  assert(h_life != NULL);
  initialState(SIZE, h_life);
  printGrid(SIZE, h_life);

  char* d_life;
  cudaError_t err;
  err = cudaMalloc((void**)&d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char));
  assert(cudaSuccess == err);

  err = cudaMemcpy(d_life, h_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyHostToDevice);
  assert(cudaSuccess == err);

  int copyingBlocksRows = SIZE / THREADS;
  int copyingBlocksColumns = SIZE + 2) / THREADS;

  std::cout << "copyingBlocksRows " << copyingBlocksRows << std::endl;
  std::cout << "copyingBlocksColumns " << copyingBlocksColumns << std::endl;

  copyHaloRows<<<copyingBlocksRows, THREADS>>>(d_life, SIZE);
  copyHaloColumns<<<copyingBlocksColumns, THREADS>>>(d_life, SIZE);

  err = cudaDeviceSynchronize();
  assert(cudaSuccess == err);

  err = cudaMemcpy(h_life, d_life, (SIZE + 2) * (SIZE + 2) * sizeof(char), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == err);

  bool ret = compareLastRealAndTopHaloRow(h_life, SIZE);
  assert(ret);

  ret = compareFirstRealAndBottomHaloRow(h_life, SIZE);
  assert(ret);

  printGrid(SIZE, h_life);

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
