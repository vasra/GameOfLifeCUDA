#include <gol.cuh>
#include <assert.h>
#include <iostream>
#include <vector>

void copyHaloRowsTest();
void copyHaloColumnsTest();

constexpr int size = 840;
constexpr int threads = 512;

int
main() {
    std::cout << "Hello world test!" << std::endl;
    copyHaloRowsTest();
    copyHaloColumnsTest();
    return 0;
}

void
copyHaloRowsTest() {
    char* h_life = (char*)malloc((size + 2) * (size + 2) * sizeof(char));
    assert(h_life != NULL);
    initialState(size, h_life);

    char* d_life;
    cudaError_t err;
    err = cudaMalloc((void**)&d_life, (size + 2) * (size + 2) * sizeof(char));
    assert(err == cudaSuccess);

    err = cudaMemcpy(d_life, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    constexpr int copyingBlocksRows = size / threads;
    std::vector<char> firstRealRow(h_life + size + 3, h_life + size * 2);
    std::vector<char> lastRealRow(h_life + size * (size + 2) + 1, h_life + size * (size + 2) + size);

    assert(firstRealRow.size() == size);

    copyHaloRows<<<copyingBlocksRows, threads>>>(d_life, size);

    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

    err = cudaMemcpy(h_life, d_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    std::vector<char> topHaloRow(h_life + 1, h_life + size);
    std::vector<char> bottomHaloRow(h_life + (size + 2) * (size + 1) + 1, h_life + (size + 2) * (size + 1) + size);

    assert(firstRealRow == bottomHaloRow);
    assert(lastRealRow  == topHaloRow);

    cudaFree(d_life);
    free(h_life);
}

void copyHaloColumnsTest() {
    char* h_life = (char*)malloc((size + 2) * (size + 2) * sizeof(char));
    assert(h_life != NULL);
    initialState(size, h_life);

    char* d_life;
    cudaError_t err;
    err = cudaMalloc((void**)&d_life, (size + 2) * (size + 2) * sizeof(char));
    assert(err == cudaSuccess);

    err = cudaMemcpy(d_life, h_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    constexpr int copyingBlocksColumns = size / threads;
    std::vector<char> firstRealColumn;
    std::vector<char> lastRealColumn;

    // copy bottom-right corner element
    firstRealColumn.push_back(*(h_life + size * (size + 2) + size));

    // copy bottom-left corner element
    lastRealColumn.push_back(*(h_life + size * (size + 2) + 1));

    // copy rest of the elements
    for (int i = 1; i < size + 1; i++) {
        firstRealColumn.push_back(*(h_life + i * (size + 2) + 1));
        lastRealColumn.push_back(*(h_life + i * (size + 2) + size));
    }

    // copy top-right corner element
    firstRealColumn.push_back(*(h_life + size * 2));

    // copy top-left corner element
    lastRealColumn.push_back(*(h_life + size + 3));

    assert(firstRealColumn.size() == size + 2);
    assert(lastRealColumn.size() == size + 2);

    copyHaloRows<<<copyingBlocksColumns, threads>>>(d_life, size);

    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

    err = cudaMemcpy(h_life, d_life, (size + 2) * (size + 2) * sizeof(char), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    std::vector<char> leftHaloColumn;
    std::vector<char> rightHaloColumn;

    // copy halo columns
    for (int i = 0; i < size + 2; i++) {
        leftHaloColumn.push_back(*(h_life + i * (size + 2)));
        rightHaloColumn.push_back(*(h_life + i * (size + 2) + size + 1));
    }

    assert(size + 2 == leftHaloColumn.size());
    assert(size + 2 == rightHaloColumn.size());

    assert(firstRealColumn == rightHaloColumn);
    assert(lastRealColumn == leftHaloColumn);

    printGrid(size + 2, size + 2, h_life);
    cudaFree(d_life);
    free(h_life);
}
