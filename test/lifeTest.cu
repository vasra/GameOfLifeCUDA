#include <gol.cuh>
#include <assert.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cmath>

void copyHaloRowsTest();
void copyHaloColumnsTest();
bool compareFirstRealAndBottomHaloRow(char* h_life, int size);
bool compareLastRealAndTopHaloRow(char* h_life, int size);

// the size of the grid without the halos
constexpr double size = 25.0f;
constexpr double threads = 16.0f;

// The four corners of the grid that contain REAL elements and not halo elements
constexpr int topLeft     = size + 3;
constexpr int topRight    = topLeft + size - 1;
constexpr int bottomLeft  = (size + 2) * size + 1;
constexpr int bottomRight = bottomLeft + size - 1;

constexpr int bottomLeftHalo  = (size + 1) * (size + 2) + 1;
constexpr int bottomRightHalo = bottomLeftHalo + size - 1;

int
main() {
    std::cout << "Hello world test!" << std::endl;
    std::cout << "Top left element index     : " << topLeft << std::endl;
    std::cout << "Top right element index    : " << topRight << std::endl;
    std::cout << "Bottom left element index  : " << bottomLeft << std::endl;
    std::cout << "Bottom right element index : " << bottomRight << std::endl;
    copyHaloRowsTest();
    //copyHaloColumnsTest();
    return 0;
}

void
copyHaloRowsTest() {
  char* h_life = (char*)malloc((size + 2) * (size + 2) * sizeof(char));
  assert(h_life != NULL);
  initialState(size, h_life);
  //printGrid(size, h_life);

  char* d_life;
  cudaError_t err;
  err = cudaMalloc((void**)&d_life, (size + 2) * (size + 2) * sizeof(char));
  assert(cudaSuccess == err);

  err = cudaMemcpy(d_life, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
  assert(cudaSuccess == err);

  int copyingBlocksRows = static_cast<int>(ceil(size / threads));
  std::cout << "copyingBlocksRows " << copyingBlocksRows << std::endl;
  bool ret = compareLastRealAndTopHaloRow(h_life, size);
  ret = compareFirstRealAndBottomHaloRow(h_life, size);

  copyHaloRows<<<copyingBlocksRows, threads>>>(d_life, size);

  err = cudaDeviceSynchronize();
  assert(cudaSuccess == err);

  err = cudaMemcpy(h_life, d_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == err);
  printGrid(size, h_life);

  cudaFree(d_life);
  free(h_life);
}

// void copyHaloColumnsTest() {
//     char* h_life = (char*)malloc((size + 2) * (size + 2) * sizeof(char));
//     assert(h_life != NULL);
//     initialState(size, h_life);

//     char* d_life;
//     cudaError_t err;
//     err = cudaMalloc((void**)&d_life, (size + 2) * (size + 2) * sizeof(char));
//     assert(cudaSuccess == err);

//     err = cudaMemcpy(d_life, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
//     assert(cudaSuccess == err);

//     constexpr int copyingBlocksColumns = size / threads;
//     std::vector<char> firstRealColumn;
//     std::vector<char> lastRealColumn;

//     // copy bottom-right corner element
//     firstRealColumn.push_back(*(h_life + size * (size + 2) + size));

//     // copy bottom-left corner element
//     lastRealColumn.push_back(*(h_life + size * (size + 2) + 1));

//     // copy rest of the elements
//     for (int i = 1; i < size + 1; i++) {
//         firstRealColumn.push_back(*(h_life + i * (size + 2) + 1));
//         lastRealColumn.push_back(*(h_life + i * (size + 2) + size));
//     }

//     // copy top-right corner element
//     firstRealColumn.push_back(*(h_life + size * 2));

//     // copy top-left corner element
//     lastRealColumn.push_back(*(h_life + size + 3));

//     assert(firstRealColumn.size() == size + 2);
//     assert(lastRealColumn.size() == size + 2);

//     copyHaloRows<<<copyingBlocksColumns, threads>>>(d_life, size);

//     err = cudaDeviceSynchronize();
//     assert(cudaSuccess == err);

//     err = cudaMemcpy(h_life, d_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyDeviceToHost);
//     assert(cudaSuccess == err);

//     std::vector<char> leftHaloColumn;
//     std::vector<char> rightHaloColumn;

//     // copy halo columns
//     for (int i = 0; i < size + 2; i++) {
//         leftHaloColumn.push_back(*(h_life + i * (size + 2)));
//         rightHaloColumn.push_back(*(h_life + i * (size + 2) + size + 1));
//     }

//     assert(size + 2 == leftHaloColumn.size());
//     assert(size + 2 == rightHaloColumn.size());

//     assert(firstRealColumn == rightHaloColumn);
//     assert(lastRealColumn == leftHaloColumn);

//     printGrid(size, h_life);
//     cudaFree(d_life);
//     free(h_life);
// }

bool
compareFirstRealAndBottomHaloRow(char* h_life, int size) {
  // Indices of the first element in the first REAL row, and the
  // first element in the bottom halo row respectively. We do not take
  // into account the corner elements
  int topLeftReal = size + 3;
  int bottomLeftHalo = (size + 1) * (size + 2) + 1;

  for (int i = 0; i < size; i++) {
    if (*(h_life + i + size + 2) != *(h_life + size * (size + 2) + i)) {
      return false;
    }
  }
  return true;
}

bool
compareLastRealAndTopHaloRow(char* h_life, int size) {
  // Indices of the first element in the last REAL row, and the
  // first element in the top halo row respectively. We do not take
  // into account the corner elements
  int bottomLeftReal = size * (size + 2) + 1;
  int topLeftHalo = 1;

  for (int i = 0; i < size; i++) {
    if (*(h_life + bottomLeftReal + i) != *(h_life + topLeftHalo + i)) {
      return false;
    }
  }
  return true;
}
