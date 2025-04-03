#include <stdio.h>

__global__ void myKernelHello( void ) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	printf("Hello This is GPU thread ID: %d\n", tid);
}


	int main( void ){
		int total_work = 1024;
		int block_size = 32;
		int grid_size = total_work / block_size;
		
		dim3 dimBlock(block_size);
		dim3 dimGrid(grid_size);
	
		myKernelHello<<<dimGrid, dimBlock>>>();
		cudaDeviceSynchronize();
		
		printf("CPU_FIN\n");
}	
