#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define N(275)

static void simple_sgemm(int n, float, alpha, const float *A, float *B, float beta, float *C) {
    int i; int j; int k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            float prod = 0;
            for (k = 0; k < n; k++) {
                prod += A[k * n + i] * B[j * n + k];
            }
            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

int main(int argc, char **argv) {
    cublasStatus_t status;
    float *h_A; float *h_B;
    float *h_C; float *h_C_ref;
    float *d_A = 0;
    float *d_B = 0;
    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;

    cublasHandle_t handle;
    int dev = findCudaDevice(argc, (const char **)argv); if (dev == -1) {
        return EXIT_FAILURE;
    }

    printf("simpleCUBLAS test running...\n");
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    h_A = reinterpret_cast<float *>(malloc(n2 * sizeof(h_A[0])));

    if (h_A == 0) {
        fprintf(stderr, "!!!!host memory allocation error\n");
        return EXIT_FAILURE;
    }
    h_B = reinterpret_cast<float *>(malloc(n2 * sizeof(h_B[0]))); if (h_B == 0) {
        fprintf(stderr, "!!!!host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }
    h_C = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0]))); if (h_C == 0) {
        fprintf(stderr, "!!!!host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    for (i = 0; i < n2; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!device memory allocation error (write A)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!device memory allocation error (write B)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!device memory allocation error (write C)\n");
        return EXIT_FAILURE;
    }
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!kernel execution error.\n");
        return EXIT_FAILURE;
    }

    h_C = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0]))); if (h_C == 0) {
        fprintf(stderr, "!!!!host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!device  access error (read C)\n");
        return EXIT_FAILURE;
    }

    error_norm = 0; ref_norm = 0;
    for (i = 0; i < n2; i++) {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff; ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
    ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));
    if (fabs(ref_norm) < 1e-7) {
        fprintf(stderr, "!!!!reference norm is 0\n");
        return EXIT_FAILURE;
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    if cudaFree(d_A) != cudaSuccess) {
        fprintf(stderr, "!!!!memory free error (A)\n");
        return EXIT_FAILURE;
    }
    if cudaFree(d_B) != cudaSuccess) {
        fprintf(stderr, "!!!!memory free error (B)\n");
        return EXIT_FAILURE;
    }
    if cudaFree(d_C) != cudaSuccess) {
        fprintf(stderr, "!!!!memory free error (C)\n");
        return EXIT_FAILURE;
    }

    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!CUBLAS shutdown error\n");
        return EXIT_FAILURE;
    }
    if (error_norm / ref_norm < 1e-6f) {
        printf("simpleCUBLAS test passed\n");
        exit(EXIT_SUCCESS);
    } else {
        printf("simpleCUBLAS test failed\n");
        exit(EXIT_FAILURE);
    }
}