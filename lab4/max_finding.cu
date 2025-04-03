#include <stdio.h>

__global__ static void timeReduction(const float *input, float *output, clock_t *timer) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_offset = 2 * blockDim.x * blockIdx.x;

    if (tid == 0)
        timer[bid] = clock();

    shared[tid] = input[global_offset + tid];
    shared[tid + blockDim.x] = input[global_offset + tid + blockDim.x];

    for (int d = blockDim.x; d > 0; d /= 2) {
        __syncthreads();

        if (tid < d) {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 > f0) {
                shared[tid] = f1;
            }
        }
    }

    if (tid == 0)
        output[bid] = shared[0];

        __syncthreads();

        if (tid == 0)
            timer[bid + gridDim.x] = clock();
}

int main(int argc, char **argv) {
    printf("CUDA Clock sample\n");
    const int threads_per_blk = 256;
    const int number_of_blocks = 64;

    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    clock_t timer[number_of_blocks * 2];
    float CPUinput[16384 * 2];
    float result[256];

    for (int i = 0; i < 16384 * 2; i++) {
        CPUinput[i] = (float)i;
    }
    clock_t start, end;

    cudaMalloc((void **)&dinput, sizeof(float) * 16384 * 2);
    cudaMalloc((void **)&doutput, sizeof(float) * number_of_blocks);
    cudaMalloc((void **)&dtimer, sizeof(clock_t) * number_of_blocks * 2);

    cudaMemcpy(dinput, CPUinput, sizeof(float)*16384*2, cudaMemcpyHostToDevice);

    dim3 dimGird(number_of_blocks);
    dim3 dimBlock(threads_per_blk);

    start = clock();
    timeReduction<<<dimGird, dimBlock, sizeof(float)*2*threads_per_blk>>>(dinput, doutput, dtimer);
    end = clock();

    cudaMemcpy(timer, dtimer, sizeof(clock_t) * number_of_blocks * 2, cudaMemcpyHostToDevice);
    const int number_of_return = 16384*2 / (256*2);
    cudaMemcpy(result, doutput, sizeof(float) * number_of_return, cudaMemcpyHostToDevice);

    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dtimer);

    long double avgElapsedClocks = 0;

    for (int i = 0; i < number_of_blocks; i++) {
        avgElapsedClocks += (long double)(timer[i + number_of_blocks] - timer[i]);
    }

    avgElapsedClocks = avgElapsedClocks / number_of_blocks;
    printf("Average clock/block = %Lf\n", avgElapsedClocks);
    printf("GPU time = %f Second\n", (float)(end - start)/CLOCKS_PER_SEC);

    start = clock();
    float max = 0.0;
    for (int i = 0; i < 16384*2; i++) {
        if (CPUinput[i] > max) max = CPUinput[i];
    }

    end = clock();
    printf("CPU time = %f Second\n", (float)(end - start)/CLOCKS_PER_SEC);

    return EXIT_SUCCESS;
}