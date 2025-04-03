#include <cstdio>
__global__ void myKernelHello(void){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    printf("[%d, %d] - thread-ID:%d\n", blockIdx.x, threadIdx.x, tid);
}

int main() {
    cudaDeviceProp props;
    printf("\tdevice\n");
    cudaGetDeviceProperties(&props, 0);
    printf("\t\tname: %s\n", props.name);

    return 0;
}