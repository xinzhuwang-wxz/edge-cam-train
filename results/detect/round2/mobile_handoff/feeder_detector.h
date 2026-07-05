// Feeder 检测器 · 移动端共用核心（iOS Obj-C++ / Android JNI 都包这一份）
// 依赖：ncnn。模型 nanodet_feeder5_mobile_416.{param,bin}（sigmoid+归一化已焊入）。
// 用法：load() 一次；每帧 detect(bgr, w, h) → vector<Detection>（原图像素坐标）。
#pragma once
#include <string>
#include <vector>
#include "net.h"  // ncnn

struct Detection {
    int   label;              // 0..4，见 label_name()
    float score;              // 置信度 0..1
    float x1, y1, x2, y2;     // 框，原图像素坐标（已缩放回输入图尺寸）
};

class FeederDetector {
public:
    // param/bin：ncnn 模型路径。use_fp16：移动端建议 true（体积/内存减半、近无损）。
    // 返回 0 成功，非 0 失败。
    int load(const char* param_path, const char* bin_path, bool use_fp16 = true);

    // bgr：BGR 排列、0-255、任意尺寸的连续像素（width*height*3）。内部 resize 到 416，
    // 模型已焊归一化，无需自己做 mean/std。conf/nms 阈值可调。
    std::vector<Detection> detect(const unsigned char* bgr, int width, int height,
                                  float conf_thresh = 0.40f, float nms_thresh = 0.50f);

    static const char* label_name(int i);  // 0=bird 1=squirrel 2=cat 3=person 4=other_animal

private:
    ncnn::Net net_;
};
