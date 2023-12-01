#include <stdio.h>
#include <cuda_runtime.h>

#define COLUMNS 3
#define ROWS 2

// CUDA Kernel Device code
__global__ void add(int *a, int *b, int *c) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int i = (COLUMNS * y) + x;
    c[i] = a[i] + b[i];
}

// Function to allocate and initialize host arrays
void initializeHostArrays(int a[ROWS][COLUMNS], int b[ROWS][COLUMNS]) {
    for (int y = 0; y < ROWS; y++)
        for (int x = 0; x < COLUMNS; x++) {
            a[y][x] = x;
            b[y][x] = y;
        }
}

int main() {
    int a[ROWS][COLUMNS], b[ROWS][COLUMNS], c[ROWS][COLUMNS];

    // Initialize host arrays
    initializeHostArrays(a, b);

    // Allocate and copy host arrays to device
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, ROWS * COLUMNS * sizeof(int));
    cudaMalloc((void **)&dev_b, ROWS * COLUMNS * sizeof(int));
    cudaMalloc((void **)&dev_c, ROWS * COLUMNS * sizeof(int));

    cudaMemcpy(dev_a, a, ROWS * COLUMNS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, ROWS * COLUMNS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    dim3 grid(COLUMNS, ROWS);
    add<<<grid, 1>>>(dev_a, dev_b, dev_c);

    // Copy the result back to the host
    cudaMemcpy(c, dev_c, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToHost);

    // Output Arrays
    for (int y = 0; y < ROWS; y++) {
        for (int x = 0; x < COLUMNS; x++) {
            printf("[%d][%d]=%d ", y, x, c[y][x]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
