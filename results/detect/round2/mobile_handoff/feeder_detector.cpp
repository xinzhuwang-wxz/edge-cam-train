// Feeder 检测器核心实现。decode 规格照抄 NanoDet 官方后处理，已用 Python 参考在测试集验证
// （pred-vs-GT IoU 0.89~0.99）。输出 [3598,37] = [5 类概率(已 sigmoid), 32 框分布(DFL: 4 边×8 bin)]。
#include "feeder_detector.h"
#include <algorithm>
#include <cmath>

namespace {
const char* kLabels[] = {"bird", "squirrel", "cat", "person", "other_animal"};
const int   kStrides[] = {8, 16, 32, 64};   // 4 个 FPN level
const int   kInput    = 416;                // 训练输入方形（keep_ratio=false，直接拉伸）
const int   kNumClass = 5;
const int   kRegMax   = 7;                   // 每条边 reg_max+1 = 8 个 bin

void nms(std::vector<Detection>& dets, float nms_thresh) {
    std::stable_sort(dets.begin(), dets.end(),   // stable：平票按原(anchor)序，与 Python 参考一致
                     [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<char> removed(dets.size(), 0);
    std::vector<Detection> out;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(dets[i]);
        const float ai = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j] || dets[j].label != dets[i].label) continue;
            const float xx1 = std::max(dets[i].x1, dets[j].x1);
            const float yy1 = std::max(dets[i].y1, dets[j].y1);
            const float xx2 = std::min(dets[i].x2, dets[j].x2);
            const float yy2 = std::min(dets[i].y2, dets[j].y2);
            const float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
            const float inter = w * h;
            const float aj = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            if (inter / (ai + aj - inter + 1e-9f) > nms_thresh) removed[j] = 1;
        }
    }
    dets.swap(out);
}
}  // namespace

const char* FeederDetector::label_name(int i) {
    return (i >= 0 && i < kNumClass) ? kLabels[i] : "?";
}

int FeederDetector::load(const char* param_path, const char* bin_path, bool use_fp16) {
    net_.opt.use_fp16_storage    = use_fp16;   // 移动端建议开：体积/内存减半、近无损
    net_.opt.use_fp16_arithmetic = use_fp16;
    // net_.opt.use_vulkan_compute = true;      // 需 GPU 加速时开启（并 net_.set_vulkan_device）
    if (net_.load_param(param_path)) return -1;
    if (net_.load_model(bin_path))   return -2;
    return 0;
}

std::vector<Detection> FeederDetector::detect(const unsigned char* bgr, int W, int H,
                                              float conf, float nms_thresh) {
    // 预处理：resize 到 416（直接拉伸，与训练一致）。模型已焊 (x-mean)/std，勿再归一化。
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr, ncnn::Mat::PIXEL_BGR, W, H, kInput, kInput);
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;               // [h=3598, w=37]
    ex.extract("out0", out);

    const float sx = static_cast<float>(W) / kInput;
    const float sy = static_cast<float>(H) / kInput;
    std::vector<Detection> dets;

    int idx = 0;
    for (int si = 0; si < 4; ++si) {                 // level 顺序 [8,16,32,64]，须与模型一致
        const int stride = kStrides[si];
        const int fh = (kInput + stride - 1) / stride, fw = fh;
        for (int y = 0; y < fh; ++y) {               // 行主序：y 外 x 内
            for (int x = 0; x < fw; ++x, ++idx) {
                const float* row = out.row(idx);     // 37 个值
                int cls = 0; float best = row[0];
                for (int c = 1; c < kNumClass; ++c)
                    if (row[c] > best) { best = row[c]; cls = c; }
                if (best < conf) continue;

                // DFL decode：每条边 8 bin → softmax → ·[0..7] → ·stride
                float dist[4];
                for (int e = 0; e < 4; ++e) {
                    const float* p = row + kNumClass + e * (kRegMax + 1);
                    float mx = p[0];
                    for (int k = 1; k <= kRegMax; ++k) if (p[k] > mx) mx = p[k];
                    float sum = 0.f, acc = 0.f;
                    for (int k = 0; k <= kRegMax; ++k) { float ev = std::exp(p[k] - mx); sum += ev; acc += ev * k; }
                    dist[e] = (acc / sum) * stride;
                }
                const float cx = x * stride, cy = y * stride;   // 中心 = x*stride（无 +0.5）
                Detection d;
                d.label = cls; d.score = best;
                d.x1 = std::max(0.f, std::min((float)W, (cx - dist[0]) * sx));
                d.y1 = std::max(0.f, std::min((float)H, (cy - dist[1]) * sy));
                d.x2 = std::max(0.f, std::min((float)W, (cx + dist[2]) * sx));
                d.y2 = std::max(0.f, std::min((float)H, (cy + dist[3]) * sy));
                dets.push_back(d);
            }
        }
    }
    nms(dets, nms_thresh);
    return dets;
}
