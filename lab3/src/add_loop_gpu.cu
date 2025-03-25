#include <stdio.h>
#include "../common/book.h"

#define MAX_N 10000

__global__ void add(int *a, int *b, int *c, int N, int offset) {
    int tid = blockIdx.x;
    if (tid < N/2)
        c[tid + offset] = a[tid + offset] + b[tid + offset];
}

int main(void) {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    for(int N = 1000; N <= MAX_N; N += 1000) {
        a = (int*)malloc(N * sizeof(int));
        b = (int*)malloc(N * sizeof(int));
        c = (int*)malloc(N * sizeof(int));

        HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

        for (int i = 0; i < N; i++) {
            a[i] = -i;
            b[i] = i*i;
        }

        HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
        
        if (N == MAX_N) {
            clock_t start = clock();
            add<<<N/2, 1>>>(dev_a, dev_b, dev_c, N, 0);  // 앞쪽 절반 처리
            add<<<N/2, 1>>>(dev_a, dev_b, dev_c, N, N/2);  // 뒤쪽 절반 처리
            clock_t end = clock();
            printf("N=%d (두 번 나누어 실행)일 때 소요 시간: %lf\n", N, (double)(end - start) / CLOCKS_PER_SEC);
        } else {
            clock_t start = clock();
            add<<<N, 1>>>(dev_a, dev_b, dev_c, N, 0);
            clock_t end = clock();
            printf("N=%d일 때 소요 시간: %lf\n", N, (double)(end - start) / CLOCKS_PER_SEC);
        }

        HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));
        
        HANDLE_ERROR(cudaFree(dev_a));
        HANDLE_ERROR(cudaFree(dev_b));
        HANDLE_ERROR(cudaFree(dev_c));
        
        free(a);
        free(b);
        free(c);
    }

    return 0;
}