#include "../common/book.h"
#define MAX_N 10000

void add(int *a, int *b, int *c, int N) {
    int tid = 0;
    while(tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main(void) {
    for(int N = 1000; N <= MAX_N; N += 1000) {
        int *a = (int*)malloc(N * sizeof(int));
        int *b = (int*)malloc(N * sizeof(int));
        int *c = (int*)malloc(N * sizeof(int));

        for (int i = 0; i < N; i++) {
            a[i] = -i;
            b[i] = i*i;
        }

        clock_t start = clock();
        add(a, b, c, N);
        clock_t end = clock();
        printf("N=%d일 때 소요 시간: %lf\n", N, (double)(end - start) / CLOCKS_PER_SEC);

        free(a);
        free(b);
        free(c);
    }

    return 0;
}