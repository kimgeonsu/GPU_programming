#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>

#define N (2) // 행렬 크기

int main() {
    // cuBLAS 핸들 초기화
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 호스트 메모리 할당 및 초기화
    float h_A[N * N] = {5.0f, 2.0f, 2.0f, 3.0f}; // A 행렬 [[5, 2], [2, 3]]
    float h_b[N] = {-2.0f, 4.0f};               // b 벡터 [-2, 4]
    float h_x[N] = {0.0f, 0.0f};                // 초기 해 x = [0, 0]

    // 디바이스 메모리 할당
    float *d_A, *d_b, *d_x, *d_r, *d_p, *d_Ap;
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_r, N * sizeof(float));
    cudaMalloc((void **)&d_p, N * sizeof(float));
    cudaMalloc((void **)&d_Ap, N * sizeof(float));

    // 호스트 데이터를 디바이스로 복사
    cublasSetMatrix(N, N, sizeof(float), h_A, N, d_A, N);
    cublasSetVector(N, sizeof(float), h_b, 1, d_b, 1);
    cublasSetVector(N, sizeof(float), h_x, 1, d_x, 1);

    // CGM 알고리즘 변수 초기화
    float alpha, beta, rTr, tmp;
    int k = 0, maxit = 1000;
    float epsilon = 1e-6f;

    // r = b - A * x
    float minusOne = -1.0f, zero = 0.0f, one = 1.0f;
    cublasSgemv(handle, CUBLAS_OP_N, N, N, &minusOne, d_A, N, d_x, 1, &one, d_b, 1); // r = b - A * x
    cublasScopy(handle, N, d_b, 1, d_r, 1);                                         // r = b

    // p = r
    cublasScopy(handle, N, d_r, 1, d_p, 1);

    // rTr = r^T * r
    cublasSdot(handle, N, d_r, 1, d_r, 1, &rTr);

    // CGM 반복
    while (k < maxit) {
        // A * p 계산
        cublasSgemv(handle, CUBLAS_OP_N, N, N, &one, d_A, N, d_p, 1, &zero, d_Ap, 1);

        // alpha = (r^T * r) / (p^T * A * p)
        cublasSdot(handle, N, d_p, 1, d_Ap, 1, &tmp);
        if (tmp == 0.0f) {
            printf("Division by zero detected in alpha calculation.\n");
            break;
        }
        alpha = rTr / tmp;

        // x = x + alpha * p
        cublasSaxpy(handle, N, &alpha, d_p, 1, d_x, 1);

        // r = r - alpha * A * p
        tmp = -alpha;
        cublasSaxpy(handle, N, &tmp, d_Ap, 1, d_r, 1);

        // 새로운 r^T * r 계산
        cublasSdot(handle, N, d_r, 1, d_r, 1, &tmp);

        // 수렴 조건 확인
        if (sqrt(tmp) < epsilon) break;

        // beta = (새로운 r^T * r) / (이전 r^T * r)
        beta = tmp / rTr;
        rTr = tmp;

        // p = r + beta * p
        cublasSscal(handle, N, &beta, d_p, 1);
        cublasSaxpy(handle, N, &one, d_r, 1, d_p, 1);

        k++;
    }

    // 결과를 호스트로 복사
    cublasGetVector(N, sizeof(float), d_x, 1, h_x, 1);

    // 결과 출력
    printf("Conjugate Gradient Method 결과 (x):\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_x[i]);
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);

    // cuBLAS 핸들 종료
    cublasDestroy(handle);

    return 0;
}