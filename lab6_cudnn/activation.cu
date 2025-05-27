#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>


int main(int argc, char** argv) {
    cudnnHandle_t my_handler;
    cudnnCreate(&my_handler);

    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 1, c = 1, h = 1, w = 21;
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
        host_data[i] = (i - 10) * 1.00f;
    }
    
    // CPU에서 GPU로 데이터 복사
    cudaMemcpy(data, host_data, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Original array: ");
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        printf("%.6f| ", host_data[i]);
    }
    printf("\n");
    float alpha[1] = {1};
    float beta[1] = {0.0};

    // Clipped RELU 활성화 함수 설정
    cudnnActivationDescriptor_t clipped_relu_activation;
    cudnnCreateActivationDescriptor(&clipped_relu_activation);
    
    // 다양한 coefficient 값으로 Clipped RELU 테스트
    float coeffs[] = {0.0f, 1.0f, -1.0f};
    const char* coeff_names[] = {"0.0", "1.0", "-1.0"};
    
    for(int coeff_idx = 0; coeff_idx < 3; coeff_idx++) {
        cudnnSetActivationDescriptor(clipped_relu_activation,
                                   CUDNN_ACTIVATION_CLIPPED_RELU,
                                   CUDNN_NOT_PROPAGATE_NAN,
                                   coeffs[coeff_idx]);  // coefficient value

        printf("\nClipped ReLU with coefficient %s:\n", coeff_names[coeff_idx]);
        
        // Forward pass
        cudnnActivationForward(my_handler,
                             clipped_relu_activation,
                             alpha,
                             data_desc,
                             data,
                             beta,
                             data_desc,
                             data);

        // GPU에서 CPU로 결과 복사
        cudaMemcpy(host_data, data, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);
        
        printf("Clipped RELU result: ");
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            printf("%.6f| ", host_data[i]);
        }
        printf("\n");
    }

    // 메모리 해제
    cudaFree(data);
    free(host_data);
    cudnnDestroyTensorDescriptor(data_desc);
    cudnnDestroyActivationDescriptor(clipped_relu_activation);
    cudnnDestroy(my_handler);
    
    return 0;
}