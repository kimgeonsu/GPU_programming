#include <cstdio>

__global__ void thread_space(void) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("[%d, %d] - thread - ID:%d\n", blockIdx.x, threadIdx.x, tid);
}

int main() {
    int total_work = 1024;
    int block_size = 8;
    int grid_size = total_work / block_size;
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);
    thread_space<<<dimGrid, dimBlock>>>();
    // cudaDeviceSynchronize();
    printf("CPU_FIN\n");

    return 0;
}