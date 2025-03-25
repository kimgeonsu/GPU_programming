#include <stdio.h>
#include "../common/book.h"

#define N 10000

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main( void ) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    HANDLE_ERROR( cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR( cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR( cudaMalloc((void**)&dev_c, N * sizeof(int)));

    for (int i=0; i < N; i++) {
        a[i] = -i;
        b[i] = i*i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
    clock_t start = clock();
    add<<<N, 1>>>(dev_a, dev_b, dev_c);
    clock_t end = clock();
    printf("소요 시간:%lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    HANDLE_ERROR(cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice));
    // for (int i =0; i<N;i++) {
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return 0;
}