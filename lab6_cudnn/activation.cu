#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>


int main(int argc, char** argv) {
    cudnnHandle_t my_handler;
    cudnnCreate(&my_handler);

    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 1, c = 1, h = 1, w = 10;
    int NUM_ELEMENTS = n * c * h * w;

    cudnnTensorDescriptor_t data_desc;
    cudnnCreateTensorDescriptor(&data_desc);
    cudnnSetTensor4dDescriptor(data_desc, format, dtype, n, c, h, w);
    float* data;
    float* host_data;  // CPU 메모리 추가
    
    // CPU 메모리 할당
    host_data = (float*)malloc(NUM_ELEMENTS * sizeof(float));
    
    // GPU 메모리 할당
    cudaMalloc((void**)&data, NUM_ELEMENTS * sizeof(float));
    
    // 초기 데이터 설정
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        host_data[i] = i * 1.00f;
    }
    
    // CPU에서 GPU로 데이터 복사
    cudaMemcpy(data, host_data, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Original array:");
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        printf("%.6f| ", host_data[i]);
    }
    printf("\n");
    float alpha[1] = {1};
    float beta[1] = {0.0};

    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(my_handler, sigmoid_activation, alpha, data_desc, data, beta, data_desc, data);

    // GPU에서 CPU로 결과 복사
    cudaMemcpy(host_data, data, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Sigmoid result:");
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        printf("%.6f| ", host_data[i]);
    }
    printf("\n");

    // 메모리 해제
    cudaFree(data);           // GPU 메모리 해제
    free(host_data);          // CPU 메모리 해제
    cudnnDestroyTensorDescriptor(data_desc);
    cudnnDestroyActivationDescriptor(sigmoid_activation);
    cudnnDestroy(my_handler);
    
    return 0;
}