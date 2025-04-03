#include <cstdio>

int main() {
    cudaDeviceProp props;

    printf("\tdevice\n");
    cudaGetDeviceProperties(&props, 0);
    printf("\t\tname: %s\n", props.name);
    printf("\t\tmultiProcessorCount: %d\n", props.multiProcessorCount);
    printf("\t\tmaxThreadsPerBlock: %d\n", props.maxThreadsPerBlock);
    printf("\t\ttotalGlobalMem: %lu\n", props.totalGlobalMem);
    printf("\t\tsharedMemPerBlock: %lu\n", props.sharedMemPerBlock);

    return 0;
}