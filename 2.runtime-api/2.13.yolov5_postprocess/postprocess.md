
# YOLOv5 后处理

YOLOv5 的输出 tensor（n x 85）：cx, cy, width, height, classification*80

把 PyTorch 的数据转换成 numpy 后，用 tobytes 转成二进制写到文件，再用 C++ 读取

总共有两大步
- 第一步：对YOLOv5 的输出进行解码，将框恢复出来
- 第二步：nms，根据条件去除重合度较高的框，每张图里面，每个类别只保留一个框


