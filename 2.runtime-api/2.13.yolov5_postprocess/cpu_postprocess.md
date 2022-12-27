
# YOLOv5 后处理

YOLOv5 的输出 tensor（n x 85）：cx, cy, width, height, objectness, classification*80

把 PyTorch 的数据转换成 numpy 后，用 tobytes 转成二进制写到文件，再用 C++ 读取

总共有两大步
- 第一步：对YOLOv5 的输出进行解码，将框恢复出来
- 第二步：nms，根据条件去除重合度较高的框，每张图里面，每个类别只保留一个框

>头文件
```c++
#include <stdio.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
```

&emsp;
# 1 load file
- 读取二进制文件（ios::in | ios::binary）
- 通过 seekg(0, ios::end) 移动指针，通过 tellg() 获得文件大小
- 通过 read() 函数将数据读取到 vector<uint8_t> 中
- 因为是二进制文件，uint8_t 为一个 bytes，所以返回值为 vector<uint8_t> 类型

```c++
static vector<uint8_t> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if(!in.is_open()) return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    vector<uint8_t> load_data;
    if(length > 0){
        load_data.resize(length);
        in.seekg(0, ios::beg);
        // 这里 &load_data[0] = load_data.data()
        in.read((char*)load_data.data(), length);
    }
    in.close();
    return load_data;
}
```


&emsp;
# 2 decode
- 构建 Box 结构体
- 获取这张图片里面 box 的数量，因为每个框的数据是 85 个 float：cx, cy, width, height, objectness, classification*80，所以：nBoxes = datalength/(sizeof(float)*85)
- 为了减少不必要计算，提升性能，先判断 objectness 是否小于 confidence_threshold，用 if...continue 跳过 ojectness 过小的 Box
- 同理，先找 80 个类别的概率值，得出 confidence，用 if...continue 跳过 confidence 过膝小的 Box

```c++
static vector<Box> cpu_decode(vector<uint8_t>& load_data, float confidence_threshold=0.25f){
    int nBoxes = load_data.size() / (sizeof(float)*85);

    float* pBoxes = (float*)load_data.data();
    vector<Box> decode_res;
    for(int i=0; i<nBoxes; i++)
    {
        float* pBox = pBoxes + i*85;
        float objectness = pBox[4];
        if(objectness < confidence_threshold) continue;
        // 获取 label 和 confidence
        float* pClasses = pBox+5; // 或者 &pBox[5]
        int label = std::max_element(pClasses, pClasses+80) - pClasses;
        float probability = pClasses[label];
        float confidence  = objectness * probability;
        if(confidence < confidence_threshold) continue;

        // 获取 left, top, right, bottom
        float cx = pBox[0];
        float cy = pBox[1];
        float width  = pBox[2];
        float height = pBox[3];

        float left = cx - width*0.5f;
        float top  = cy - height*0.5f;
        float right  = cx + width*0.5f;
        float bottom = cy + width*0.5f;
        decode_res.emplace_back(left, top, right, bottom, confidence, label);
    }
    return decode_res;
}
```


&emsp;
# 3 nms（标准）
- 对 decode 出来的框进行排序，用匿名函数比较 confidence 的大小
- 用 lambda 函数写计算 iou 的函数
- 用两层循环进行遍历
    - 设置 remove_flag，size 和 nBoxes 一样，向后添加 remove_flag
    - 在循环中，用 if...continue 和 remove_flag 节省计算量
    - 第1层循环为：`i=0; i<nBoxes`
        - 判断是否被添加过 remove_flag，有即跳过
        - 否则 emplace_back 这个 box
    - 第2层循环为：`j=i+1; j<nBoxes`，跳过之前对比过的 box
        - 判断是否被添加过 remove_flags，有即跳过
        - 判断是否是相同 label 的框，不是则跳过
        - 计算 iou，大于阈值的，将第2层循环里的 box 添加 remove_flag
    
```c++
static vector<Box> cpu_nms(vector<Box> decode_box, float nms_threshold=0.45f)
{
    std::sort(
        decode_box.begin(), decode_box.end(),
        [](Box& a, Box& b){return a.confidence > b.confidence;}
    );

    auto iou = [](Box& a, Box& b){
        float cross_left = std::max(a.left, b.left);
        float cross_top  = std::max(a.top, b.top);
        float cross_right  = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right-cross_left) *
                           std::max(0.0f, cross_bottom-cross_top);
        float union_area = std::max(0.0f, a.right-a.left)*std::max(0.0f, a.bottom-a.top) + 
                           std::max(0.0f, b.right-b.left)*std::max(0.0f, b.bottom-b.top) -
                           cross_area;
        if(cross_area==0 || union_area==0) return 0.0f;
        return cross_area / union_area;
    };

    vector<Box> nms_res;
    int nBoxes = decode_box.size();
    vector<bool> remove_flag(nBoxes);
    for(int i=0; i<nBoxes; i++)
    {
        if(remove_flag[i]) continue;
        Box ibox = decode_box[i];
        nms_res.emplace_back(ibox);

        for(int j=i+1; j<nBoxes; j++)
        {
            if(remove_flag[j]) continue;
            Box jbox = decode_box[j];
            if(ibox.label == jbox.label)
            {
                if(iou(ibox, jbox) > nms_threshold)
                    remove_flag[j]=true;
            }
        }
    }
    return nms_res;
}
```