#include "../common/book.h"
#define MAX_N 10000

void add(int *a, int *b, int *c, int N, int offset) {
    int tid = 0;
    while(tid < N/2) {
        c[tid + offset] = a[tid + offset] + b[tid + offset];
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

        if (N == MAX_N) {
            clock_t start = clock();
            add(a, b, c, N, 0);      // 앞쪽 절반 처리
            add(a, b, c, N, N/2);    // 뒤쪽 절반 처리
            clock_t end = clock();
            printf("N=%d (두 번 나누어 실행)일 때 소요 시간: %lf\n", N, (double)(end - start) / CLOCKS_PER_SEC);
        } else {
            clock_t start = clock();
            add(a, b, c, N, 0);
            clock_t end = clock();
            printf("N=%d일 때 소요 시간: %lf\n", N, (double)(end - start) / CLOCKS_PER_SEC);
        }

        free(a);
        free(b);
        free(c);
    }

    return 0;
}