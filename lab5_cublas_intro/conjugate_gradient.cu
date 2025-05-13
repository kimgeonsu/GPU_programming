#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    const int n = 5;
    const int rows = n;

    double host_A[25] = {5,2,3,1, -10,
                         7,3,2,-4,1,
                         5,3,9,1,2,
                         2,1,-2,8,2,
                         9,3,1,-9,3
                        };
    double host_b[5] = {10, 2, 3, -4, 5};
    double host_x[5] = {0, 0, 0, 0, 0};

    double *A, *b, *x, *r, *p, *Axp;
    cudaMalloc((void**)&A, n * n * sizeof(double));
    cudaMalloc((void**)&b, n * sizeof(double));
    cudaMalloc((void**)&x, n * sizeof(double));
    cudaMalloc((void**)&r, n * sizeof(double));
    cudaMalloc((void**)&p, n * sizeof(double));
    cudaMalloc((void**)&Axp, n * sizeof(double));
    
    cudaMemcpy(A, host_A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x, host_x, n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double zero = 0.0, one = 1.0, minusOne = -1.0;
    double alpha = 0.0, beta = 0.0, rxr = 0.0, tmp = 0.0;
    double epsilon = 1e-6;
    int maxit = 100000000;

    cublasDcopy(handle, n, b, 1, r, 1);

    cublasDgemv(handle, CUBLAS_OP_N, n, n, &minusOne, A, rows, x, 1, &one, r, 1);

    cublasDcopy(handle, n, r, 1, p, 1);
    cublasDdot(handle, n, r, 1, r, 1, &rxr);

    int k = 0;
    while (k < maxit) {
        cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, A, rows, p, 1, &zero, Axp, 1); // Axp = A * p
        cublasDdot(handle, n, p, 1, Axp, 1, &tmp);
        alpha = rxr / tmp;

        cublasDaxpy(handle, n, &alpha, p, 1, x, 1);
        tmp = -alpha;
        cublasDaxpy(handle, n, &tmp, Axp, 1, r, 1); // r = r - alpha * Axp
        cublasDdot(handle, n, r, 1, r, 1, &tmp);

        if (sqrt(tmp) < epsilon) {
            break;
        }

        beta = tmp / rxr;
        rxr = tmp;
        cublasDscal(handle, n, &beta, p, 1); // p = beta * p
        cublasDaxpy(handle, n, &one, r, 1, p, 1);

        k++;
    }
    cudaMemcpy(host_x, x, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    int i = 0;
    for (i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, host_x[i]);
    }

    cudaFree(A);
    cudaFree(b);
    cudaFree(x);
    cudaFree(r);
    cudaFree(p);
    cudaFree(Axp);
    cublasDestroy(handle);
    return 0;
}