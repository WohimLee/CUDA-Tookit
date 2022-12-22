#include <cuda_runtime.h>
#include <stdio.h>
#define min(a, b) ((a) < (b) ? (a) : (b))


typedef unsigned char uint8_t;

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

struct Pixel{
    int x=0, y=0;
    uint8_t* data;
    float c0, c1, c2;
};


struct SrcPoint{
    float x=0.0, y=0.0;
    // 周围的4个像素点
    Pixel p1; // left top
    Pixel p2; // right top
    Pixel p3; // left bottom
    Pixel p4; // right bottom
    float w1, w2, w3, w4;
};

struct AffineMatrix{
    float i2d[6];
    float d2i[6];

    void getInvertMatrix(float imat[6], float omat[6]){
        // 原矩阵 imat
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];
        //        0 ,                   0 ,                   1

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D; // A00
        float A22 = i00 * D; // A11
        float A12 = -i01 * D; // A01
        float A21 = -i10 * D; // A10
        float b1 = -A11 * i02 - A12 * i12; // A20
        float b2 = -A21 * i02 - A22 * i12; // A21
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }

    void compute(const GPUData srcGPU, const GPUData dstGPU){
        float scale_x = dstGPU.width  / (float)srcGPU.width;
        float scale_y = dstGPU.height / (float)srcGPU.height;

        float scale = min(scale_x, scale_y);
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = 
            -scale * srcGPU.width  * 0.5  + dstGPU.width * 0.5 + scale * 0.5 - 0.5;

        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = 
            -scale * srcGPU.height * 0.5 + dstGPU.height * 0.5 + scale * 0.5 - 0.5;

        getInvertMatrix(i2d, d2i);
    }

};

__device__ void affine_project(float* matrix, Pixel& dstPoint, SrcPoint& srcPoint){
    srcPoint.x = matrix[0]*dstPoint.x + matrix[1]*dstPoint.y + matrix[2];
    srcPoint.y = matrix[3]*dstPoint.x + matrix[4]*dstPoint.y + matrix[5];
}


__global__ void warpaffine_bilinear_kernel(
    GPUData srcGPU, GPUData dstGPU, 
    AffineMatrix matrix, uint8_t fill_value
){
    // printf("%d x %d\n", srcGPU.height, srcGPU.width);
    Pixel dstPoint; SrcPoint srcPoint;
    dstPoint.x = blockDim.x * blockIdx.x + threadIdx.x;
    dstPoint.y = blockDim.y * blockIdx.y + threadIdx.y;
    if(dstPoint.x >= dstGPU.width || dstPoint.y >= dstGPU.height) return;
    // dstPoint 通过逆矩阵投影找到 srcPoint
    affine_project(matrix.d2i, dstPoint, srcPoint);
    // 开始做双线性差值
    dstPoint.c0 = fill_value;
    dstPoint.c1 = fill_value;
    dstPoint.c2 = fill_value;
    // 上面先给 dstPoint 3 个通道默认值
    if(srcPoint.x < -1 || srcPoint.x >= srcGPU.width || 
       srcPoint.y < -1 || srcPoint.y >= srcGPU.height){
        // 保证只处理原图上的像素点
    }else{

        int coord_x_low = floorf(srcPoint.x);
        int coord_y_low = floorf(srcPoint.y);
        int coord_x_high = coord_x_low + 1;
        int coord_y_high = coord_y_low + 1;
        // srcPoint 周围的4个像素确定坐标
        srcPoint.p1.x=coord_x_low;  srcPoint.p1.y=coord_y_low;
        srcPoint.p2.x=coord_x_high; srcPoint.p2.y=coord_y_low;
        srcPoint.p3.x=coord_x_low;  srcPoint.p3.y=coord_y_high;
        srcPoint.p4.x=coord_x_high; srcPoint.p4.y=coord_y_high;

        // 计算每个像素 x, y 方向上的比例
        float portion_x_low = srcPoint.x - coord_x_low;
        float portion_y_low = srcPoint.y - coord_y_low;
        float portion_x_high = 1 - portion_x_low;
        float portion_y_high = 1 - portion_y_low;
        
        // 计算权重
        srcPoint.w1 = portion_x_high * portion_y_high;
        srcPoint.w2 = portion_x_low  * portion_y_high;
        srcPoint.w3 = portion_x_high * portion_y_low;
        srcPoint.w4 = portion_x_low  * portion_y_low;
        
        // 下面的 4 个 if 语句都是为了保证只对在 src 上的像素取值
        if(srcPoint.p1.x >= 0 || srcPoint.p1.y >= 0)
            srcPoint.p1.data = srcGPU.data + srcPoint.p1.y * srcGPU.row_size + srcPoint.p1.x*3;

        if(srcPoint.p2.x <= srcGPU.width || srcPoint.p2.y >= 0)
            srcPoint.p2.data = srcGPU.data + srcPoint.p2.y * srcGPU.row_size + srcPoint.p2.x*3;

        if(srcPoint.p3.x >= 0 || srcPoint.p3.y <= srcGPU.height)
            srcPoint.p3.data = srcGPU.data + srcPoint.p3.y * srcGPU.row_size + srcPoint.p3.x*3;

        if(srcPoint.p4.x <= srcGPU.width || srcPoint.p4.y <= srcGPU.height)
            srcPoint.p4.data = srcGPU.data + srcPoint.p4.y * srcGPU.row_size + srcPoint.p4.x*3;

        dstPoint.c0 = floorf(
            srcPoint.w1*srcPoint.p1.data[0] + srcPoint.w2*srcPoint.p2.data[0] + srcPoint.w3*srcPoint.p3.data[0] + srcPoint.w4*srcPoint.p4.data[0]);
        dstPoint.c1 = floorf(
            srcPoint.w1*srcPoint.p1.data[1] + srcPoint.w2*srcPoint.p2.data[1] + srcPoint.w3*srcPoint.p3.data[1] + srcPoint.w4*srcPoint.p4.data[1]);
        dstPoint.c2 = floorf(
            srcPoint.w1*srcPoint.p1.data[2] + srcPoint.w2*srcPoint.p2.data[2] + srcPoint.w3*srcPoint.p3.data[2] + srcPoint.w4*srcPoint.p4.data[2]);
    }
    dstPoint.data = dstGPU.data + dstPoint.y*dstGPU.row_size + dstPoint.x*3;
    dstPoint.data[0] = dstPoint.c0;
    dstPoint.data[1] = dstPoint.c1;
    dstPoint.data[2] = dstPoint.c2;
    // printf("(%d, %d) c0=%d, c1=%d, c2=%d\n", dstPoint.x, dstPoint.y, dstPoint.c0, dstPoint.c1, dstPoint.c2);
}

void lauch_kernel(
    GPUData& srcGPU, GPUData& dstGPU, uint8_t fill_value)
{
    dim3 block_size(32, 32);
    dim3 grid_size((dstGPU.width+31)/32, (dstGPU.height+31)/32);
    AffineMatrix affine;
    affine.compute(srcGPU, dstGPU);
    warpaffine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        srcGPU, dstGPU, 
        affine, fill_value
    );
}
