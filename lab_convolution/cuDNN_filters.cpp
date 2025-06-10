#include <cudnn.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

#define checkCUDNN(expr) do { \
    cudnnStatus_t status = (expr); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "Error on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // 입력 텐서 설정 (1x1x7x7)
    cudnnTensorDescriptor_t input_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 7, 7));

    // Convolution Descriptor 생성
    cudnnConvolutionDescriptor_t conv_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 필터 Descriptor 생성 (크기는 아래 루프에서 설정)
    cudnnFilterDescriptor_t filter_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));

    // 출력 텐서 Descriptor (크기는 동적으로 설정)
    cudnnTensorDescriptor_t output_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));

    // 입력 데이터 초기화 (1x1x7x7)
    const int input_size = 7;
    float input[input_size*input_size];
    for (int i = 0; i < input_size*input_size; ++i) input[i] = static_cast<float>(i + 1);

    std::cout << "\n\n-=-=-=-=-=-" << input_size << "x" << input_size << " INPUT"<<"-=-=-=-=-=-" <<std::endl;
    for (int j = 0; j < input_size; ++j) {
        for (int k = 0; k < input_size; ++k)
            std::cout << input[j * input_size + k] << "\t";
        std::cout << std::endl;
    }
    std::cout<<std::endl;

    float *d_input, *d_output = nullptr, *d_filter;
    cudaMalloc(&d_input, sizeof(input));
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);

    void* workspace = nullptr;
    size_t workspace_size = 0;

    // 필터 크기, stride, dilation 설정
    int array_filter_size[3] = {3, 5, 3};
    int array_strides[3]      = {1, 2, 1};
    int array_dilations[3]    = {1, 1, 2};


    for (int loop = 0; loop < 3; loop++) 
    {
        int filter_size = array_filter_size[loop];
        int stride = array_strides[loop];
        int dilation = array_dilations[loop];
        int pad = ((filter_size - 1) * dilation) / 2;

        // 필터 설정
        checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,1, 1, filter_size, filter_size));

        // Convolution 설정
        checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        // 출력 텐서 크기 계산
        int n, c, h, w;
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w));

        // 출력 텐서 Descriptor 재설정
        checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

        // 출력 메모리 재할당
        if (d_output != nullptr) 
            cudaFree(d_output);
        cudaMalloc(&d_output, sizeof(float) * n * c * h * w);

        // 필터 weight: box blur (평균 필터)
        std::vector<float> filter_weights(filter_size * filter_size, 1.0f / (filter_size * filter_size));
        cudaMalloc(&d_filter, sizeof(float) * filter_size * filter_size);
        cudaMemcpy(d_filter, filter_weights.data(), sizeof(float) * filter_size * filter_size, cudaMemcpyHostToDevice);

        // 알고리즘 직접 지정 (cuDNN 8 이상 호환)
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        // workspace 크기 확인 및 할당
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
        cudaMalloc(&workspace, workspace_size);

        // Convolution 수행
        const float alpha = 1.0f, beta = 0.0f;
        checkCUDNN(cudnnConvolutionForward(cudnn,
            &alpha, input_desc, d_input,
            filter_desc, d_filter,
            conv_desc, algo, workspace, workspace_size,
            &beta, output_desc, d_output));

        // 결과 복사 및 출력
        std::vector<float> output(n * c * h * w);
        cudaMemcpy(output.data(), d_output, sizeof(float) * output.size(), cudaMemcpyDeviceToHost);
        std::cout << "=== " << filter_size << "x" << filter_size << " 필터 / stride=" << stride << " / dilation=" << dilation << " 결과 ===\n";
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < w; ++k)
                std::cout << output[j * w + k] << "\t";
            std::cout << std::endl;
        }
        std::cout<<std::endl;

        cudaFree(d_filter);
        cudaFree(workspace);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}