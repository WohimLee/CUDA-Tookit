# GPU Post Process
>头文件
- 包含的头文件
- CUDA 错误检查
- Box 结构体
```c++
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line)
{
    if(code != cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_str  = cudaGetErrorString(code);
        printf(
            "CUDA Runtime Error [%s:%d] %s failed. \n Error Name: %s, Error String: %s\n",
            file, line, op, err_name, err_str
        );
        return false;
    }
    return true;
}

struct Box{
    float left, top, right, bottom;
    float confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label):
        left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};
```

&emsp;
# main 函数


&emsp;
# load file
```c++
static vector<uint8_t> load_file(const string& file)
{
    ifstream in(file, ios::in | ios::binary);
    if(!in.is_open())
        return {};
    in.seekg(0, ios::end);
    size_t size = in.tellg(); // sizeof(uint8_t) = 1 Byte
    vector<uint8_t> load_data;
    if(size > 0){
        load_data.resize(size);
        in.seekg(0, ios::beg);
        in.read((char*)load_data.data(), size);
    }
    in.close();
    return load_data;
}
```

&emsp;
# malloc and launch
- 因为后面要用 cudaMemcpyAsync，所以 Host 的 output 内存必须要用 cudaMallocHost
- 使用 stream 注意一定要用 cudaStreamSynchronize()，否则会出错，核函数里面的 printf 也不会显示
- 必须是 cudaStreamSynchronize() 过后才能对
```c++
vector<Box> malloc_and_call_launch(vector<uint8_t>& load_data)
{
    hostInput.data = (float*)load_data.data();
    hostInput.size = load_data.size();
    int MAX_BOXES = 1000;
    int DECODED_BOX_ELEMENTS = 7; // l, t, r, b, confidence, label, keepflag
    deviceInput.size  = hostInput.size;
    // 前面留一个 float 用来计数
    deviceOutput.size = sizeof(float) + MAX_BOXES*DECODED_BOX_ELEMENTS*sizeof(float);
    hostOutput.size   = deviceOutput.size;

    checkRuntime(cudaMalloc(&deviceInput.data, deviceInput.size));
    checkRuntime(cudaMalloc(&deviceOutput.data, deviceOutput.size));
    checkRuntime(cudaMallocHost(&hostOutput.data, hostOutput.size));

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    checkRuntime(cudaMemcpyAsync(
        deviceInput.data, hostInput.data, deviceInput.size, 
        cudaMemcpyHostToDevice, stream
    ));
    launch_kernels(
        deviceInput, deviceOutput,
        MAX_BOXES, DECODED_BOX_ELEMENTS, stream
    );
    checkRuntime(cudaMemcpyAsync(
        hostOutput.data, deviceOutput.data, deviceOutput.size,
        cudaMemcpyDeviceToHost, stream
    ));
    checkRuntime(cudaStreamSynchronize(stream));

    vector<Box> fast_nms_res;
    int nActualBoxes = min((int)hostOutput.data[0], MAX_BOXES);
    for(int i=0; i<nActualBoxes; i++)
    {
        float* pBox = 1 + hostOutput.data + i*DECODED_BOX_ELEMENTS;
        int keep_flag = pBox[6];
        if(keep_flag){
            fast_nms_res.emplace_back(
                pBox[0], pBox[1], pBox[2], pBox[3],
                pBox[4], (int)pBox[5]
            );
        }
    }

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(deviceInput.data));
    checkRuntime(cudaFree(deviceOutput.data));
    checkRuntime(cudaFreeHost(hostOutput.data));

    // return {};
    return fast_nms_res;
}
```

&emsp;
# launch kernels
```c++

void launch_kernels(
    DeviceData& deviceInput, DeviceData& deviceOutput, 
    int MAX_BOXES, int DECODED_BOX_ELEMENTS, 
    cudaStream_t stream, 
    float confidence_threshold=0.25f, float nms_threshold=0.45f
){
    int nBoxes   = deviceInput.size / (sizeof(float)*85);
    int nClasses = 80;
    auto block = nBoxes > 32 ? 32 : nBoxes;
    auto grid  = (nBoxes + block - 1) / block;

    printf("Launch decode kernels...\n");
    decode_kernel<<<grid, block, 0, stream>>>(
        deviceInput.data, deviceOutput.data, 
        nBoxes, nClasses, MAX_BOXES, DECODED_BOX_ELEMENTS, 
        confidence_threshold
    );

    block = MAX_BOXES > 32 ? 32 : MAX_BOXES;
    grid  = (MAX_BOXES + block - 1) / block;

    printf("Launch nms kernels...\n");
    fast_nms_kernel<<<grid, block, 0, stream>>>(
        deviceOutput.data, 
        MAX_BOXES, DECODED_BOX_ELEMENTS,
        nms_threshold
    );
}
```

&emsp;
# decode kernel

```c++
static __global__ void decode_kernel(
    float* pLoadedBoxes, float* pDecodedBoxes, 
    int nBoxes, int nClasses, int MAX_BOXES, int DECODED_BOX_ELEMENTS, 
    float confidence_threshold
){

    int boxIdx = blockDim.x*blockIdx.x + threadIdx.x;
    if(boxIdx >= nBoxes) return;

    float* pLoadedBox = pLoadedBoxes + boxIdx*85;
    float objectness = pLoadedBox[4];
    if(objectness < confidence_threshold) return;
    float* pClasses  = pLoadedBox + 5;
    float confidence = 0;
    float label = 0;
    for(int i=0; i<nClasses; i++){
        // confidence = pClasses[i];
        if(pClasses[i] > confidence){
            confidence = pClasses[i];
            label = i;
        }
    }

    confidence *= objectness;
    if(confidence < confidence_threshold) return;

    int index = atomicAdd(pDecodedBoxes, 1);
    // printf("index=%d\n", index);
    if(index >= MAX_BOXES){
        printf("index=%d, >= MAX_BOXES\n");
        return;
    };
    float cx = pLoadedBox[0];
    float cy = pLoadedBox[1];
    float width  = pLoadedBox[2];
    float height = pLoadedBox[3];

    float left = cx - width*0.5f;
    float top  = cy - height*0.5f;
    float right  = cx + width*0.5f;
    float bottom = cy + height*0.5f;

    float* pDecodedBox = 1 + pDecodedBoxes + index*DECODED_BOX_ELEMENTS;
    pDecodedBox[0] = left;
    pDecodedBox[1] = top;
    pDecodedBox[2] = right;
    pDecodedBox[3] = bottom;
    pDecodedBox[4] = confidence;
    pDecodedBox[5] = label;
    pDecodedBox[6] = 1; // 1=true, 0=false
    // printf("Decoded Box %d, confidence=%f.\n", index, pDecodedBox[4]);
}
```

&emsp;
# fast nms kernel
- 所有 decode 出来的框，每个框一个线程
- 每个线程内，遍历所有的框，用以下条件筛选，节省计算
    - 跟本线程重复的框跳过
    - 只处理同 label 的框，不同的跳过（nms 和 fast nms 的区别）
    - confidence 相等的跳过
- 计算 iou，如果大于阈值，将本线程的 box 的 flag 设置为 0(ignore) 
    - 这里注意：跟 cpu 的做法不一样，不能去修改被对比的框，因为是多线程并发，有可能会多个线程同时去修改一处内存，造成冲突

```c++
static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom
){
    float cross_left    = max(aleft, bleft);
    float cross_top     = max(atop, btop);
    float cross_right   = min(aright, bright);
    float cross_bottom  = min(abottom, bbottom);
    // printf("(%f,%f,%f,%f)\n", cross_left, cross_top, cross_right, cross_bottom);
    float cross_area = max(0.0f, cross_right - cross_left)*max(0.0f, cross_bottom-cross_top);
    if(cross_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft)*max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft)*max(0.0f, bbottom - btop);
    return cross_area / (a_area + b_area - cross_area);
}

static __global__ void fast_nms_kernel(
    float* pDecodedBoxes, 
    int MAX_BOXES, int DECODED_BOX_ELEMENTS, 
    float nms_threshold
){
    int boxIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int nActualBoxes = min((int)*pDecodedBoxes, MAX_BOXES);
    if(boxIdx >= nActualBoxes) return;

    // left, top, right, bottom, confidence, class, keepflag
    float* pBox_a = 1 + pDecodedBoxes + boxIdx*DECODED_BOX_ELEMENTS;
    for(int i=0; i<nActualBoxes; i++){
        float* pBox_b = 1 + pDecodedBoxes + i*DECODED_BOX_ELEMENTS; 
        // float* pBox_b = pBox_a + DECODED_BOX_ELEMENTS; 

        // 本来是一个框或者不同类别的框，跳过
        if(i == boxIdx || pBox_a[5] != pBox_b[5]) continue;
        // 如果比当前线程的 box 的 confidence 高
        // printf("a.confidence=%f, b.confidence=%f\n", pBox_a[4], pBox_b[4]);
        if(pBox_b[4] >= pBox_a[4]){
            if(pBox_b[4] == pBox_a[4] && i < boxIdx)
                continue;
            float iou = box_iou(
                pBox_a[0], pBox_a[1], pBox_a[2], pBox_a[3],
                pBox_b[0], pBox_b[1], pBox_b[2], pBox_b[3]
            );
            // printf("iou=%f\n", iou);

            if(iou > nms_threshold){
                pBox_a[6] = 0; // 1=keep, 0=ignore
                printf("Box %d, keep flag=%d.\n", boxIdx, pBox_a[6]);
                return;
            }
        }
    }
}
```
