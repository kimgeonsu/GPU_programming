/**
* Simple MNIST CUDNN Implementation
* 
* This version implements the basic architecture:
* - 2 convolution layers (1->32->64)  
* - 2 max pooling layers
* - 1 fully connected layer (3136->10)
* - Uses weights from data_simple/ directory
*/

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <dirent.h>
#include <algorithm>
#include <vector>
#include <functional>
#include <iomanip>
#include <cmath>

#include <cuda.h>
#include <cudnn.h>

#include <FreeImage.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "gemv.h"
#include "error_util.h"

#define IMAGE_H 28
#define IMAGE_W 28

const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";

// Weight files for the simple 2-conv layer architecture
const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";

// Fully connected layer weight files
const char *fc1_bin = "fc1.bin";
const char *fc1_bias_bin = "fc1.bias.bin";

#define EXIT_WAIVED 0

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("models/data_simple/") + std::string(fname));
}

// Need the map, since scaling factor is of float type in half precision
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};
template <> struct ScaleFactorTypeMap<half1>  { typedef float Type;};

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
    template <class T>
    value_type operator()(T x) {return value_type(x);}
    value_type operator()(half1 x) {return value_type(cpu_half2float(x));}
};

template <>
class Convert<half1>
{
public:
    template <class T>
    half1 operator()(T x) {return cpu_float2half_rn (T(x));} 
    half1 operator()(half1 x) {return x;}
};

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }
    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }
    // conversion
    Convert<value_type> fromReal;
    for (int i = 0; i < size; i++)
    {
        data_h[i] = fromReal(data_tmp[i]);
    }
    delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    *data_h = new value_type[size];

    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc((void**)data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    FatalError(zMessage);
}

// Forward declaration for basic preprocessing
template <class value_type>
void applyBasicPreprocessing(value_type* imgData, bool quiet);

template <class value_type>
void readImage(const char* fname, value_type* imgData_h, bool quiet = false)
{
    // declare a host image object for an 8-bit grayscale image
    std::string sFilename(fname);
    if (!quiet) {
        std::cout << "Loading image " << sFilename << std::endl;
    }
    // Take care of half precision
    Convert<value_type> fromReal;
    
    // load gray-scale image from disk    
    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());

    // no signature? try to guess the file format from the file extension
    if (eFormat == FIF_UNKNOWN)
    {
        eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
    }

    if (eFormat == FIF_UNKNOWN)
    {
        FatalError("Unknown image format");
    }
    // check that the plugin has reading capabilities ...

    FIBITMAP *pBitmap;
    if (FreeImage_FIFSupportsReading(eFormat))
    {
        pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
    }

    if (pBitmap == 0)
    {
        FatalError("Error reading image");
    }
    
    // Check image type and convert to grayscale if needed
    bool is_color = false;
    FIBITMAP *grayBitmap = pBitmap;
    
    if (FreeImage_GetColorType(pBitmap) != FIC_MINISBLACK || FreeImage_GetBPP(pBitmap) != 8)
    {
        if (!quiet) {
            std::cout << "Color image detected - converting to grayscale..." << std::endl;
            std::cout << "Original: " << FreeImage_GetBPP(pBitmap) << " bits, " 
                      << "Color type: " << FreeImage_GetColorType(pBitmap) << std::endl;
        }
        
        // Convert to 8-bit grayscale
        grayBitmap = FreeImage_ConvertToGreyscale(pBitmap);
        if (grayBitmap == NULL) {
            FreeImage_Unload(pBitmap);
            FatalError("Failed to convert image to grayscale");
        }
        is_color = true;
        
        if (!quiet) {
            std::cout << "Converted to: " << FreeImage_GetBPP(grayBitmap) << " bits grayscale" << std::endl;
        }
    }

    int width = FreeImage_GetWidth(grayBitmap);
    int height = FreeImage_GetHeight(grayBitmap);
    
    if (width != IMAGE_W || height != IMAGE_H)
    {
        FatalError("Image dimensions missmatch");
    }
    
    // Normalize image to be in range [0,1] and apply same normalization as PyTorch
    // PyTorch normalization: (pixel/255 - 0.1307) / 0.3081
    for (int i = 0; i < height; ++i)
    { 
        unsigned char *pSrcLine = FreeImage_GetScanLine(grayBitmap, height - i - 1);
        for (int j = 0; j < width; j++)
        {
            int idx = IMAGE_W*i + j;
            float normalized_pixel = (*(pSrcLine + j) / 255.0f - 0.1307f) / 0.3081f;
            imgData_h[idx] = fromReal(normalized_pixel);
        }
    }

    // Clean up memory
    if (is_color) {
        FreeImage_Unload(grayBitmap);  // Free the converted grayscale image
    }
    FreeImage_Unload(pBitmap);         // Free the original image
    
    // Apply basic preprocessing
    if (!quiet) {
        std::cout << "Applying basic preprocessing..." << std::endl;
    }
    applyBasicPreprocessing(imgData_h, quiet);
    if (!quiet) {
        std::cout << "Basic preprocessing completed." << std::endl;
    }
}

// Basic preprocessing functions
template <class value_type>
void checkAndInvertColors(value_type* imgData, int width, int height) {
    // Check if image has white background and dark digits (needs inversion)
    
    // Method 1: Check edge pixels vs center pixels
    float edge_sum = 0.0f;
    int edge_count = 0;
    float center_sum = 0.0f;
    int center_count = 0;
    
    // Sample edge pixels (border)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i < 3 || i >= height-3 || j < 3 || j >= width-3) {
                // Edge pixels
                edge_sum += (float)imgData[i * width + j];
                edge_count++;
            } else if (i >= height/4 && i < 3*height/4 && j >= width/4 && j < 3*width/4) {
                // Center pixels
                center_sum += (float)imgData[i * width + j];
                center_count++;
            }
        }
    }
    
    float edge_avg = edge_count > 0 ? edge_sum / edge_count : 0.0f;
    float center_avg = center_count > 0 ? center_sum / center_count : 0.0f;
    
    // Method 2: Check overall brightness distribution
    float overall_sum = 0.0f;
    for (int i = 0; i < width * height; i++) {
        overall_sum += (float)imgData[i];
    }
    float overall_avg = overall_sum / (width * height);
    
    // Decision logic: Invert if background appears to be bright (stricter criteria)
    bool should_invert = false;
    
    // More strict criteria: both conditions should suggest white background
    // Condition 1: Overall brightness is significantly high
    bool high_overall_brightness = (overall_avg > 0.5f);
    
    // Condition 2: Edge is much brighter than center (strong white background signal)
    bool bright_edge_pattern = (edge_avg > center_avg + 0.5f);
    
    // Condition 3: Very high overall brightness (clearly white background)
    bool very_high_brightness = (overall_avg > 0.7f);
    
    // Invert only if we have strong evidence of white background
    if (very_high_brightness || (high_overall_brightness && bright_edge_pattern)) {
        should_invert = true;
    }
    
    if (should_invert) {
        std::cout << "Detected white background - inverting colors..." << std::endl;
        std::cout << "Edge avg: " << edge_avg << ", Center avg: " << center_avg << ", Overall avg: " << overall_avg << std::endl;
        
        // Invert all pixel values
        for (int i = 0; i < width * height; i++) {
            float val = (float)imgData[i];
            // For normalized range around [-1, 1], invert by negating
            imgData[i] = Convert<value_type>()(-val);
        }
        
        std::cout << "Color inversion completed." << std::endl;
    } else {
        std::cout << "Colors appear normal (dark background detected)." << std::endl;
        std::cout << "Edge avg: " << edge_avg << ", Center avg: " << center_avg << ", Overall avg: " << overall_avg << std::endl;
    }
}

template <class value_type>
void gaussianSmooth(value_type* imgData, int width, int height) {
    value_type* smoothed = new value_type[width * height];
    
    // 3x3 Gaussian kernel (lighter smoothing)
    float kernel[9] = {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        sum += kernel[(ki+1)*3 + (kj+1)] * (float)imgData[ni * width + nj];
                    }
                }
            }
            
            smoothed[i * width + j] = Convert<value_type>()(sum);
        }
    }
    
    // Copy back
    for (int i = 0; i < width * height; i++) {
        imgData[i] = smoothed[i];
    }
    
    delete[] smoothed;
}

template <class value_type>
void normalizeSize(value_type* imgData, int width, int height) {
    // Find bounding box of non-zero pixels
    int min_x = width, max_x = -1, min_y = height, max_y = -1;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if ((float)imgData[i * width + j] > 0.1f) {
                min_x = std::min(min_x, j);
                max_x = std::max(max_x, j);
                min_y = std::min(min_y, i);
                max_y = std::max(max_y, i);
            }
        }
    }
    
    if (max_x > min_x && max_y > min_y) {
        int digit_width = max_x - min_x + 1;
        int digit_height = max_y - min_y + 1;
        
        // Calculate scale to fit within 20x20 pixels (leaving 4-pixel border)
        float scale = std::min(20.0f / digit_width, 20.0f / digit_height);
        
        if (scale < 1.0f) {
            value_type* resized = new value_type[width * height];
            for (int i = 0; i < width * height; i++) {
                resized[i] = value_type(0.0f);
            }
            
            int new_width = (int)(digit_width * scale);
            int new_height = (int)(digit_height * scale);
            int offset_x = (width - new_width) / 2;
            int offset_y = (height - new_height) / 2;
            
            // Simple nearest neighbor scaling
            for (int i = 0; i < new_height; i++) {
                for (int j = 0; j < new_width; j++) {
                    int src_x = min_x + (int)(j / scale);
                    int src_y = min_y + (int)(i / scale);
                    int dst_x = offset_x + j;
                    int dst_y = offset_y + i;
                    
                    if (dst_x >= 0 && dst_x < width && dst_y >= 0 && dst_y < height) {
                        resized[dst_y * width + dst_x] = imgData[src_y * width + src_x];
                    }
                }
            }
            
            // Copy back
            for (int i = 0; i < width * height; i++) {
                imgData[i] = resized[i];
            }
            
            delete[] resized;
        }
    }
}

template <class value_type>
void centerOfMass(value_type* imgData, int width, int height) {
    // Calculate center of mass
    float sum_x = 0.0f, sum_y = 0.0f, total_mass = 0.0f;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float val = (float)imgData[i * width + j];
            if (val > 0.0f) {
                sum_x += j * val;
                sum_y += i * val;
                total_mass += val;
            }
        }
    }
    
    if (total_mass > 0) {
        float center_x = sum_x / total_mass;
        float center_y = sum_y / total_mass;
        
        // Calculate shift needed to center the digit
        float target_x = width / 2.0f;
        float target_y = height / 2.0f;
        int shift_x = (int)round(target_x - center_x);
        int shift_y = (int)round(target_y - center_y);
        
        // Limit shift to prevent going out of bounds
        shift_x = std::max(-width/4, std::min(width/4, shift_x));
        shift_y = std::max(-height/4, std::min(height/4, shift_y));
        
        if (abs(shift_x) > 1 || abs(shift_y) > 1) {
            // Create shifted image
            value_type* temp_img = new value_type[width * height];
            for (int i = 0; i < width * height; i++) {
                temp_img[i] = value_type(0.0f);
            }
            
            // Apply shift
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int new_i = i + shift_y;
                    int new_j = j + shift_x;
                    
                    if (new_i >= 0 && new_i < height && new_j >= 0 && new_j < width) {
                        temp_img[new_i * width + new_j] = imgData[i * width + j];
                    }
                }
            }
            
            // Copy back
            for (int i = 0; i < width * height; i++) {
                imgData[i] = temp_img[i];
            }
            
            delete[] temp_img;
        }
    }
}

template <class value_type>
void applyBasicPreprocessing(value_type* imgData, bool quiet) {
    if (!quiet) std::cout << "Applying basic preprocessing..." << std::endl;
    
    // Apply basic preprocessing steps in conservative order
    if (!quiet) checkAndInvertColors(imgData, IMAGE_W, IMAGE_H);  // Check and invert colors if needed
    else {
        // Silent version for batch processing with stricter criteria
        float overall_sum = 0.0f;
        float edge_sum = 0.0f;
        int edge_count = 0;
        float center_sum = 0.0f;
        int center_count = 0;
        
        for (int i = 0; i < IMAGE_H; i++) {
            for (int j = 0; j < IMAGE_W; j++) {
                int idx = i * IMAGE_W + j;
                float val = (float)imgData[idx];
                overall_sum += val;
                
                if (i < 3 || i >= IMAGE_H-3 || j < 3 || j >= IMAGE_W-3) {
                    edge_sum += val;
                    edge_count++;
                } else if (i >= IMAGE_H/4 && i < 3*IMAGE_H/4 && j >= IMAGE_W/4 && j < 3*IMAGE_W/4) {
                    center_sum += val;
                    center_count++;
                }
            }
        }
        
        float overall_avg = overall_sum / (IMAGE_W * IMAGE_H);
        float edge_avg = edge_count > 0 ? edge_sum / edge_count : 0.0f;
        float center_avg = center_count > 0 ? center_sum / center_count : 0.0f;
        
        // Apply same strict criteria as in verbose mode
        bool high_overall_brightness = (overall_avg > 0.5f);
        bool bright_edge_pattern = (edge_avg > center_avg + 0.5f);
        bool very_high_brightness = (overall_avg > 0.7f);
        
        if (very_high_brightness || (high_overall_brightness && bright_edge_pattern)) {
            for (int i = 0; i < IMAGE_W * IMAGE_H; i++) {
                imgData[i] = Convert<value_type>()(-(float)imgData[i]);
            }
        }
    }
    
    gaussianSmooth(imgData, IMAGE_W, IMAGE_H);    // Light noise reduction
    normalizeSize(imgData, IMAGE_W, IMAGE_H);     // Size normalization  
    centerOfMass(imgData, IMAGE_W, IMAGE_H);      // Center the digit
    
    if (!quiet) std::cout << "Basic preprocessing completed." << std::endl;
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
    typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    Convert<real_type> toReal;
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << toReal(vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

typedef enum {
        FP16_HOST  = 0, 
        FP16_CUDA  = 1,
        FP16_CUDNN = 2
 } fp16Import_t;

template <class value_type>
struct Layer_t
{
    fp16Import_t fp16Import;
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0), fp16Import(FP16_HOST){};
    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        fp16Import = _fp16Import;
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }
    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }
private:
    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy<value_type>(fname, size, data_h, data_d);
    }
};

template <class value_type>
class simple_network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type ScaleType;
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnActivationDescriptor_t activDesc;
    cublasHandle_t cublasHandle;
    
    // Performance optimization: Pre-allocated memory pools
    struct MemoryPool {
        value_type *imgData_d;
        value_type *conv1_out, *relu1_out, *pool1_out;
        value_type *conv2_out, *relu2_out, *pool2_out;
        value_type *fc1_out, *softmax_out;
        bool initialized;
        
        MemoryPool() : imgData_d(nullptr), conv1_out(nullptr), relu1_out(nullptr), 
                      pool1_out(nullptr), conv2_out(nullptr), relu2_out(nullptr),
                      pool2_out(nullptr), fc1_out(nullptr), softmax_out(nullptr), 
                      initialized(false) {}
                      
        ~MemoryPool() {
            cleanup();
        }
        
        void cleanup() {
            if (imgData_d) { cudaFree(imgData_d); imgData_d = nullptr; }
            if (conv1_out) { cudaFree(conv1_out); conv1_out = nullptr; }
            if (relu1_out) { cudaFree(relu1_out); relu1_out = nullptr; }
            if (pool1_out) { cudaFree(pool1_out); pool1_out = nullptr; }
            if (conv2_out) { cudaFree(conv2_out); conv2_out = nullptr; }
            if (relu2_out) { cudaFree(relu2_out); relu2_out = nullptr; }
            if (pool2_out) { cudaFree(pool2_out); pool2_out = nullptr; }
            if (fc1_out) { cudaFree(fc1_out); fc1_out = nullptr; }
            if (softmax_out) { cudaFree(softmax_out); softmax_out = nullptr; }
            initialized = false;
        }
        
        void initialize() {
            if (initialized) return;
            
            // Pre-allocate all memory based on network architecture
            checkCudaErrors(cudaMalloc(&imgData_d, IMAGE_H * IMAGE_W * sizeof(value_type)));
            checkCudaErrors(cudaMalloc(&conv1_out, 32 * 28 * 28 * sizeof(value_type)));  // Conv1 output
            checkCudaErrors(cudaMalloc(&relu1_out, 32 * 28 * 28 * sizeof(value_type)));  // ReLU1 output
            checkCudaErrors(cudaMalloc(&pool1_out, 32 * 14 * 14 * sizeof(value_type)));  // Pool1 output
            checkCudaErrors(cudaMalloc(&conv2_out, 64 * 14 * 14 * sizeof(value_type)));  // Conv2 output
            checkCudaErrors(cudaMalloc(&relu2_out, 64 * 14 * 14 * sizeof(value_type)));  // ReLU2 output
            checkCudaErrors(cudaMalloc(&pool2_out, 64 * 7 * 7 * sizeof(value_type)));    // Pool2 output
            checkCudaErrors(cudaMalloc(&fc1_out, 10 * sizeof(value_type)));              // FC1 output
            checkCudaErrors(cudaMalloc(&softmax_out, 10 * sizeof(value_type)));          // Softmax output
            
            initialized = true;
        }
    } memPool;
    
    // Performance optimization: GPU warmup flag
    bool warmupDone;

public:
    simple_network_t() : warmupDone(false)
    {
        checkCudaErrors( cudnnCreate(&cudnnHandle) );
        checkCudaErrors( cublasCreate(&cublasHandle) );
        
        checkCudaErrors( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCudaErrors( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCudaErrors( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCudaErrors( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCudaErrors( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCudaErrors( cudnnCreatePoolingDescriptor(&poolingDesc) );
        checkCudaErrors( cudnnCreateActivationDescriptor(&activDesc) );
        
        // Initialize pooling descriptor (2x2 max pooling)
        checkCudaErrors( cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2) );
        
        // Initialize activation descriptor (ReLU)
        checkCudaErrors( cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0) );
        
        dataType = (sizeof(value_type) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
        tensorFormat = CUDNN_TENSOR_NCHW;
        
        // Performance optimization: Set optimal convolution algorithm
        convAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        
        // Initialize memory pool
        memPool.initialize();
    }

    ~simple_network_t()
    {
        checkCudaErrors( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCudaErrors( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCudaErrors( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCudaErrors( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCudaErrors( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCudaErrors( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCudaErrors( cudnnDestroyActivationDescriptor(activDesc) );
        
        checkCudaErrors( cudnnDestroy(cudnnHandle) );
        checkCudaErrors( cublasDestroy(cublasHandle) );
    }
    
    // Performance optimization: GPU warmup
    void warmupGPU() {
        if (warmupDone) return;
        
        // Dummy operations to warm up GPU
        value_type dummy_data[IMAGE_H * IMAGE_W];
        for (int i = 0; i < IMAGE_H * IMAGE_W; i++) {
            dummy_data[i] = static_cast<value_type>(0.5);
        }
        
        checkCudaErrors(cudaMemcpy(memPool.imgData_d, dummy_data, IMAGE_H * IMAGE_W * sizeof(value_type), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        
        warmupDone = true;
    }
    
    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc((void**)data, size*sizeof(value_type)) );
    }
    
    void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                      cudnnTensorFormat_t& tensorFormat,
                      cudnnDataType_t& dataType,
                      int n, int c, int h, int w)
    {
        const int nDims = 4;
        int dimA[nDims] = {n,c,h,w};
        int strideA[nDims] = {c*h*w, h*w, w, 1};
        checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc, dataType, nDims, dimA, strideA) );
    }
    
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);
        ScaleType alpha = ScaleType(1);
        ScaleType beta  = ScaleType(1);
        checkCUDNN( cudnnAddTensor(cudnnHandle, &alpha, biasTensorDesc, layer.bias_d, &beta, dstTensorDesc, data) );
    }
    
    void fullyConnectedForward(const Layer_t<value_type>& ip, int& n, int& c, int& h, int& w, value_type* srcData, value_type** dstData)
    {
        if (c*h*w != ip.inputs)
        {
            std::cout << "FullyConnected input size mismatch " << std::endl;
            FatalError("FullyConnected input size mismatch");
        }
        
        resize(n*ip.outputs, dstData);
        
        int dim_x = ip.inputs;
        int dim_y = ip.outputs;
        ScaleType alpha = ScaleType(1), beta = ScaleType(0);
        
        // Copy bias to output first
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        // Perform matrix multiplication: output = weight * input + bias
        gemv(cublasHandle, dim_x, dim_y, alpha, ip.data_d, srcData, beta, *dstData);
        
        h = 1; w = 1; c = dim_y;
    }
    
    void convoluteForward(const Layer_t<value_type>& conv, int& n, int& c, int& h, int& w, value_type* srcData, value_type** dstData)
    {
        cudnnConvolutionFwdAlgo_t algo;
        
        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, conv.kernel_dim, conv.kernel_dim};
        
        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc, dataType, CUDNN_TENSOR_NCHW, tensorDims, filterDimA) );
        
        const int convDims = 2;
        int padA[convDims] = {1,1};  // padding=1 for 3x3 conv
        int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t convDataType = dataType;
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT;
        }
        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, convDataType) );
        
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorDims, tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];
        
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        if (convAlgorithm < 0)
        {
            // Use the most basic algorithm that should be supported on all devices
            algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            convAlgorithm = algo;
        }
        else
        {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
        }
        
        resize(n*c*h*w, dstData);
        size_t sizeInBytes = 0;
        void* workSpace = NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo, &sizeInBytes) );
        if (sizeInBytes != 0)
        {
            checkCudaErrors( cudaMalloc(&workSpace, sizeInBytes) );
        }
        
        ScaleType alpha = ScaleType(1);
        ScaleType beta = ScaleType(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle, &alpha, srcTensorDesc, srcData, filterDesc, conv.data_d, convDesc, algo, workSpace, sizeInBytes, &beta, dstTensorDesc, *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        
        if (sizeInBytes != 0)
        {
            checkCudaErrors( cudaFree(workSpace) );
        }
    }
    
    void poolForward(int& n, int& c, int& h, int& w, value_type* srcData, value_type** dstData)
    {
        const int poolDims = 2;
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, poolDims, windowDimA, paddingA, strideA) );
        
        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc, srcTensorDesc, tensorDims, tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];
        
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        resize(n*c*h*w, dstData);
        ScaleType alpha = ScaleType(1);
        ScaleType beta = ScaleType(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle, poolingDesc, &alpha, srcTensorDesc, srcData, &beta, dstTensorDesc, *dstData) );
    }
    
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0) );
        
        resize(n*c*h*w, dstData);
        
        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        ScaleType alpha = ScaleType(1);
        ScaleType beta = ScaleType(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle, activDesc, &alpha, srcTensorDesc, srcData, &beta, dstTensorDesc, *dstData) );
    }
    
    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);
        
        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);
        
        ScaleType alpha = ScaleType(1);
        ScaleType beta = ScaleType(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, srcTensorDesc, srcData, &beta, dstTensorDesc, *dstData) );
    }
    
    // Simple CNN forward pass: Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool -> FC -> Softmax
    int classify_example_simple(const char* fname,
                               const Layer_t<value_type>& conv1,
                               const Layer_t<value_type>& conv2,
                               const Layer_t<value_type>& fc1,
                               bool quiet = false)
    {
        if (!quiet) {
            std::cout << "Processing image: " << fname << std::endl;
        }
        
        // Read image
        value_type imgData_h[IMAGE_H*IMAGE_W];
        readImage(fname, imgData_h, quiet);
        
        // Allocate GPU memory for image
        value_type *imgData_d;
        checkCudaErrors(cudaMalloc((void**)&imgData_d, IMAGE_H*IMAGE_W*sizeof(value_type)));
        checkCudaErrors(cudaMemcpy(imgData_d, imgData_h, IMAGE_H*IMAGE_W*sizeof(value_type), cudaMemcpyHostToDevice));
        
        // Initialize network dimensions
        int n = 1, c = 1, h = IMAGE_H, w = IMAGE_W;
        
        // Pointers for intermediate results
        value_type *conv1_out = NULL, *relu1_out = NULL, *pool1_out = NULL;
        value_type *conv2_out = NULL, *relu2_out = NULL, *pool2_out = NULL;
        value_type *fc1_out = NULL, *softmax_out = NULL;
        
        try {
            // ===== FIRST BLOCK: Conv1 -> ReLU -> Pool =====
            
            // Conv1: 1x28x28 → 32x28x28
            convoluteForward(conv1, n, c, h, w, imgData_d, &conv1_out);
            if (!quiet) std::cout << "Conv1 output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // ReLU1
            activationForward(n, c, h, w, conv1_out, &relu1_out);
            if (!quiet) std::cout << "ReLU1 output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // MaxPool1: 32x28x28 → 32x14x14
            poolForward(n, c, h, w, relu1_out, &pool1_out);
            if (!quiet) std::cout << "Pool1 output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // ===== SECOND BLOCK: Conv2 -> ReLU -> Pool =====
            
            // Conv2: 32x14x14 → 64x14x14
            convoluteForward(conv2, n, c, h, w, pool1_out, &conv2_out);
            if (!quiet) std::cout << "Conv2 output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // ReLU2
            activationForward(n, c, h, w, conv2_out, &relu2_out);
            if (!quiet) std::cout << "ReLU2 output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // MaxPool2: 64x14x14 → 64x7x7
            poolForward(n, c, h, w, relu2_out, &pool2_out);
            if (!quiet) std::cout << "Pool2 output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // ===== FULLY CONNECTED LAYER =====
            
            // FC1: 64*7*7=3136 → 10
            fullyConnectedForward(fc1, n, c, h, w, pool2_out, &fc1_out);
            if (!quiet) std::cout << "FC1 output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // Softmax for final probabilities
            softmaxForward(n, c, h, w, fc1_out, &softmax_out);
            if (!quiet) std::cout << "Softmax output: " << n << "x" << c << "x" << h << "x" << w << std::endl;
            
            // Get the prediction by finding the class with highest probability
            value_type output_h[10];
            checkCudaErrors(cudaMemcpy(output_h, softmax_out, 10*sizeof(value_type), cudaMemcpyDeviceToHost));
            
            int prediction = 0;
            float max_prob = (float)output_h[0];
            for (int i = 1; i < 10; i++) {
                float prob = (float)output_h[i];
                if (prob > max_prob) {
                    max_prob = prob;
                    prediction = i;
                }
            }
            
            if (!quiet) {
                std::cout << "Output probabilities: ";
                for (int i = 0; i < 10; i++) {
                    std::cout << std::fixed << std::setprecision(4) << (float)output_h[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Predicted class: " << prediction << " (probability: " << max_prob << ")" << std::endl;
            }
            
            // Cleanup GPU memory
            if (imgData_d) checkCudaErrors(cudaFree(imgData_d));
            if (conv1_out) checkCudaErrors(cudaFree(conv1_out));
            if (relu1_out) checkCudaErrors(cudaFree(relu1_out));
            if (pool1_out) checkCudaErrors(cudaFree(pool1_out));
            if (conv2_out) checkCudaErrors(cudaFree(conv2_out));
            if (relu2_out) checkCudaErrors(cudaFree(relu2_out));
            if (pool2_out) checkCudaErrors(cudaFree(pool2_out));
            if (fc1_out) checkCudaErrors(cudaFree(fc1_out));
            if (softmax_out) checkCudaErrors(cudaFree(softmax_out));
            
            return prediction;
            
        } catch (...) {
            // Cleanup on error
            if (imgData_d) checkCudaErrors(cudaFree(imgData_d));
            if (conv1_out) checkCudaErrors(cudaFree(conv1_out));
            if (relu1_out) checkCudaErrors(cudaFree(relu1_out));
            if (pool1_out) checkCudaErrors(cudaFree(pool1_out));
            if (conv2_out) checkCudaErrors(cudaFree(conv2_out));
            if (relu2_out) checkCudaErrors(cudaFree(relu2_out));
            if (pool2_out) checkCudaErrors(cudaFree(pool2_out));
            if (fc1_out) checkCudaErrors(cudaFree(fc1_out));
            if (softmax_out) checkCudaErrors(cudaFree(softmax_out));
            throw;
        }
    }
};

void displayUsage()
{
    printf( "mnistCUDNN_simple {<options>}\n");
    printf( "help                   : display this help\n");
    printf( "device=<int>           : set the device to run the sample\n");
    printf( "image=<name>           : classify specific image\n");
    printf( "batch                  : classify all .pgm files in test_image directory\n");
}

// Function to get all .pgm files from test_image directory
std::vector<std::string> getPgmFiles(const char* program_path) {
    std::vector<std::string> pgm_files;
    std::string test_image_dir = "models/test_image";
    
    DIR *dir;
    struct dirent *entry;
    
    dir = opendir(test_image_dir.c_str());
    if (dir == NULL) {
        std::cerr << "Error: Cannot open test_image directory: " << test_image_dir << std::endl;
        return pgm_files;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        std::string filename = entry->d_name;
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".pgm") {
            std::string full_path = test_image_dir + "/" + filename;
            pgm_files.push_back(full_path);
        }
    }
    
    closedir(dir);
    std::sort(pgm_files.begin(), pgm_files.end());
    return pgm_files;
}

int main(int argc, char *argv[])
{   
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        displayUsage();
        exit(EXIT_WAIVED); 
    }

    std::cout << "Simple MNIST CUDNN Implementation" << std::endl;
    std::cout << "Basic 2-Conv + 1-FC architecture" << std::endl;
    
    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
    showDevices();

    int device = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        checkCudaErrors( cudaSetDevice(device) );
    }
    std::cout << "Using device " << device << std::endl;
    
    if (checkCmdLineFlag(argc, (const char **)argv, "image"))
    {
        char* image_name;
        getCmdLineArgumentString(argc, (const char **)argv,
                                 "image", (char **) &image_name);        

        std::cout << "\nTesting simple single precision model with custom image\n";
        simple_network_t<float> mnist;
        
        // Load the simple 2-layer CNN architecture
        Layer_t<float> conv1(1, 32, 3, conv1_bin, conv1_bias_bin, argv[0]);     // 1→32
        Layer_t<float> conv2(32, 64, 3, conv2_bin, conv2_bias_bin, argv[0]);    // 32→64
        Layer_t<float> fc1(3136, 10, 1, fc1_bin, fc1_bias_bin, argv[0]);        // 64*7*7=3136→10
        
        int result = mnist.classify_example_simple(image_name, conv1, conv2, fc1);
        
        std::cout << "\nResult of simple classification: " << result << std::endl;

        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    
    // Batch processing option
    if (checkCmdLineFlag(argc, (const char **)argv, "batch"))
    {
        std::vector<std::string> pgm_files = getPgmFiles(argv[0]);
        
        if (pgm_files.empty()) {
            std::cout << "No .pgm files found in test_image directory" << std::endl;
            cudaDeviceReset();
            exit(EXIT_WAIVED);
        }
        
        std::cout << "\nProcessing " << pgm_files.size() << " .pgm files from test_image directory:\n" << std::endl;
        std::cout << "=== SIMPLE MODEL PREDICTIONS ===" << std::endl;
        
        simple_network_t<float> mnist;
        
        // Load simple 2-layer CNN architecture
        Layer_t<float> conv1(1, 32, 3, conv1_bin, conv1_bias_bin, argv[0]);
        Layer_t<float> conv2(32, 64, 3, conv2_bin, conv2_bias_bin, argv[0]);
        Layer_t<float> fc1(3136, 10, 1, fc1_bin, fc1_bias_bin, argv[0]);
        
        for (size_t i = 0; i < pgm_files.size(); i++) {
            int result = mnist.classify_example_simple(pgm_files[i].c_str(), conv1, conv2, fc1, true);
            
            // Extract just the filename from the full path
            std::string filename = pgm_files[i];
            size_t lastSlash = filename.find_last_of('/');
            if (lastSlash != std::string::npos) {
                filename = filename.substr(lastSlash + 1);
            }
            
            std::cout << filename << " -> Predicted: " << result << std::endl;
        }
        
        std::cout << "\n=== SIMPLE BATCH PROCESSING COMPLETED ===" << std::endl;
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    // Default behavior - demonstrate simple architecture
    if (argc == 1 || (argc == 2) && checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        std::cout << "\nTesting simple single precision model\n";
        simple_network_t<float> mnist;
        
        // Load simple 2-layer CNN architecture
        Layer_t<float> conv1(1, 32, 3, conv1_bin, conv1_bias_bin, argv[0]);
        Layer_t<float> conv2(32, 64, 3, conv2_bin, conv2_bias_bin, argv[0]);
        Layer_t<float> fc1(3136, 10, 1, fc1_bin, fc1_bias_bin, argv[0]);
        
        std::string image_path = std::string("models/data/") + std::string(first_image);
        
        int result = mnist.classify_example_simple(image_path.c_str(), conv1, conv2, fc1);
        
        std::cout << "Classification result: " << result << std::endl;
        
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    
    displayUsage();
    cudaDeviceReset();
    exit(EXIT_WAIVED);
} 