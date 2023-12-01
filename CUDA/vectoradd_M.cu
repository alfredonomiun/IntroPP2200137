#include <stdio.h>
#include <stdlib.h>

// CUDA Kernel Device code
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Function to allocate and initialize host vectors
void initializeHostVectors(float *&h_A, float *&h_B, float *&h_C, int numElements) {
    h_A = (float *)malloc(numElements * sizeof(float));
    h_B = (float *)malloc(numElements * sizeof(float));
    h_C = (float *)malloc(numElements * sizeof(float));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to allocate and initialize device vectors
void initializeDeviceVectors(float *&d_A, float *&d_B, float *&d_C, int numElements) {
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **)&d_A, numElements * sizeof(float));
    HANDLE_ERROR(err, "Failed to allocate device vector A");

    err = cudaMalloc((void **)&d_B, numElements * sizeof(float));
    HANDLE_ERROR(err, "Failed to allocate device vector B");

    err = cudaMalloc((void **)&d_C, numElements * sizeof(float));
    HANDLE_ERROR(err, "Failed to allocate device vector C");
}

// Function to copy host vectors to device
void copyHostToDevice(float *d_A, float *d_B, float *h_A, float *h_B, int numElements) {
    cudaError_t err = cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    HANDLE_ERROR(err, "Failed to copy vector A from host to device");

    err = cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);
    HANDLE_ERROR(err, "Failed to copy vector B from host to device");
}

// Function to copy device vector to host
void copyDeviceToHost(float *h_C, float *d_C, int numElements) {
    cudaError_t err = cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(err, "Failed to copy vector C from device to host");
}

// Function to free device memory
void freeDeviceMemory(float *d_A, float *d_B, float *d_C) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Function to free host memory
void freeHostMemory(float *h_A, float *h_B, float *h_C) {
    free(h_A);
    free(h_B);
    free(h_C);
}

// Function to reset the device
void resetDevice() {
    cudaError_t err = cudaDeviceReset();
    HANDLE_ERROR(err, "Failed to deinitialize the device");
}

int main(void) {
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A, *h_B, *h_C;
    initializeHostVectors(h_A, h_B, h_C, numElements);

    float *d_A, *d_B, *d_C;
    initializeDeviceVectors(d_A, d_B, d_C, numElements);

    copyHostToDevice(d_A, d_B, h_A, h_B, numElements);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    HANDLE_ERROR(err, "Failed to launch vectorAdd kernel");

    copyDeviceToHost(h_C, d_C, numElements);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    freeDeviceMemory(d_A, d_B, d_C);
    freeHostMemory(h_A, h_B, h_C);

    resetDevice();

    printf("Done\n");
    return 0;
}
