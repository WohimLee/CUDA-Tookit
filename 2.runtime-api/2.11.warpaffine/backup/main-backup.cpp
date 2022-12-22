#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// 宏定义一个取较小值的函数
#define min(a,b) ((a)<(b) ? (a) : (b))

// 检查运行时的错误
// 先宏定义声明，再实现
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_msg  = cudaGetErrorString(code);
        printf("CUDA runtime error[%s:%d] %s failed, Error: %s, msg: %s\n", file, line, err_name, err_msg);
        return false;
    }
    return true;
}

struct GPUData{
    int width  = 0;
    int height = 0;
    int size   = 0;
    int row_size = 0;
    uint8_t* data = nullptr;
    GPUData() = default;
    GPUData(int _height, int _width): height(_height), width(_width){
        row_size = width*3;
        size = width*height*3;
    }
};

void lauch_kernel(GPUData& srcGPU, GPUData& dstGPU, uint8_t fill_value);


Mat malloc_and_call(const Mat& srcImg, const Size& dstSize){
    Mat output(dstSize, CV_8UC3);
    GPUData srcGPU(srcImg.rows, srcImg.cols);
    GPUData dstGPU(dstSize.height, dstSize.width);
    
    checkRuntime(cudaMalloc(&srcGPU.data, srcGPU.size));
    checkRuntime(cudaMalloc(&dstGPU.data, dstGPU.size));
    checkRuntime(cudaMemcpy(srcGPU.data, srcImg.data, srcGPU.size, cudaMemcpyHostToDevice));
    
    lauch_kernel(srcGPU, dstGPU, 114);
    
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, dstGPU.data, dstGPU.size, cudaMemcpyDeviceToHost));
    checkRuntime(cudaFree(srcGPU.data));
    checkRuntime(cudaFree(dstGPU.data));
    return output;
}



int main(int argc, char** argv){
    Mat image = imread("zand.jpg");
    Mat output = malloc_and_call(image, Size(640, 640));
    imwrite("output.jpg", output);
    printf("Done. Save to output.jpg\n");
    return 0;
}