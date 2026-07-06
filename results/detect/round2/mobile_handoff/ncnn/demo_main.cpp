// 独立 demo：读图 → detect → 打印 JSONL → 画框存盘。用于集成前先验证 C++ 核心。
// 编译见 CMakeLists.txt。用法: ./feeder_demo <image> [conf] [nms] [out.jpg]
#include "feeder_detector.h"
#include <opencv2/opencv.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <image> [conf] [nms] [out.jpg]\n", argv[0]); return 1; }
    const float conf = argc > 2 ? atof(argv[2]) : 0.40f;
    const float nms  = argc > 3 ? atof(argv[3]) : 0.50f;

    cv::Mat img = cv::imread(argv[1]);   // BGR
    if (img.empty()) { printf("cannot read %s\n", argv[1]); return 1; }

    FeederDetector det;
    const bool fp16 = getenv("FP16") ? atoi(getenv("FP16")) != 0 : true;  // FP16=0 关闭(对齐 fp32 参考)
    if (det.load("nanodet_feeder5_mobile_416.param", "nanodet_feeder5_mobile_416.bin", fp16)) {
        printf("load model failed\n"); return 1;
    }
    auto dets = det.detect(img.data, img.cols, img.rows, conf, nms);

    for (auto& d : dets)
        printf("{\"label\":\"%s\",\"score\":%.4f,\"box\":[%.1f,%.1f,%.1f,%.1f]}\n",
               FeederDetector::label_name(d.label), d.score, d.x1, d.y1, d.x2, d.y2);
    fprintf(stderr, "# %zu detections\n", dets.size());

    if (argc > 4) {
        for (auto& d : dets) {
            cv::rectangle(img, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2), {0, 255, 0}, 2);
            cv::putText(img, cv::format("%s %.2f", FeederDetector::label_name(d.label), d.score),
                        cv::Point(d.x1, std::max(0.f, d.y1 - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
        }
        cv::imwrite(argv[4], img);
        fprintf(stderr, "# drawn -> %s\n", argv[4]);
    }
    return 0;
}
