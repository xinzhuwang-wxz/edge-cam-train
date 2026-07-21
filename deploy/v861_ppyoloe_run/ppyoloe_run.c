/*
 * ppyoloe_run — V861 板端 PP-YOLOE-s 一体化串行推理工具
 *
 *   输入  : JPG/PNG/BMP（awcv_imread）或 raw NV21（--nv21 WxH，相机原生格式）
 *   推理  : AWNN 静态模式（precompiler）——实例常驻，每帧 ~43ms，NPU 内存恒定 14.4MB
 *   解码  : CPU 侧 sigmoid + DFL(softmax 期望) + anchor + 逐类 NMS（对齐 decode_ref.py）
 *   输出  : <out>/results.jsonl（一行一图）+ <out>/<name>_det.jpg（标框图，可关）
 *
 * 用法:
 *   ppyoloe_run [-m model_dir] [-o out_dir] [-c conf] [-n nms] [--no-draw]
 *               [--nv21 WxH] img1 [img2 ...]
 *   默认: model_dir=model  out_dir=out  conf=0.45  nms=0.50
 *   --nv21 640x640 之后的输入按 raw NV21 解析（Y 平面 + VU 交织，w*h*3/2 字节）
 *
 * 为什么必须静态模式（血泪，见 docs/detect/05-V861-真板部署.md §6）:
 *   dynamic 模式每次 inference 都重建计算图且不还 IPU 内存 —— 第 2 帧即
 *   dma_mem_alloc fail，硬砸会挂内核。静态模式预编译一次：43ms/帧、内存恒定。
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "AWNN_interface.h"
#include "AWNN_simpleocv.h"

#define INPUT_SIZE 640
#define NUM_CLASS 5
#define REG_MAX 16 /* 17 个 DFL bin: 0..16 */
#define MAX_DETS 512
#define CV_8UC3 3 /* ★ncnn simpleocv 惯例: type=通道数(不是 OpenCV 的 16)——传 16 会段错误 */

static const char* NAMES[NUM_CLASS] = {"bird", "squirrel", "cat", "person", "other_animal"};
/* BGR 顺序（simpleocv 与 OpenCV 同为 BGR 底图） */
static const unsigned char COLORS[NUM_CLASS][3] = {
    {116, 229, 46}, /* bird 绿 */
    {28, 159, 255}, /* squirrel 橙 */
    {255, 182, 56}, /* cat 蓝 */
    {86, 86, 255},  /* person 红 */
    {230, 108, 203} /* other_animal 紫 */
};
static const int STRIDES[3] = {8, 16, 32};
static const int GRIDS[3] = {80, 40, 20};
/* 输出 blob 名: cls/reg × 3 尺度（顺序与转换时的模型一致） */
static const char* OUT_NAMES[6] = {
    "conv2d_81.tmp_0", "conv2d_84.tmp_0", /* s8  cls[5,80,80]  reg[68,80,80] */
    "conv2d_74.tmp_0", "conv2d_77.tmp_0", /* s16 cls[5,40,40]  reg[68,40,40] */
    "conv2d_67.tmp_0", "conv2d_70.tmp_0"  /* s32 cls[5,20,20]  reg[68,20,20] */
};

typedef struct {
    float x1, y1, x2, y2, score;
    int cls;
} det_t;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---------- 预处理 ---------- */

/* BGR(或RGB) u8 HWC 双线性缩放到 640x640，可选通道交换 → dst RGB u8 HWC */
static void resize_to_input(const unsigned char* src, int sw, int sh, int swap_rb,
                            unsigned char* dst) {
    const float fx = (float)sw / INPUT_SIZE, fy = (float)sh / INPUT_SIZE;
    for (int y = 0; y < INPUT_SIZE; y++) {
        float syf = (y + 0.5f) * fy - 0.5f;
        if (syf < 0) syf = 0;
        int sy0 = (int)syf;
        int sy1 = sy0 + 1 < sh ? sy0 + 1 : sh - 1;
        float wy = syf - sy0;
        for (int x = 0; x < INPUT_SIZE; x++) {
            float sxf = (x + 0.5f) * fx - 0.5f;
            if (sxf < 0) sxf = 0;
            int sx0 = (int)sxf;
            int sx1 = sx0 + 1 < sw ? sx0 + 1 : sw - 1;
            float wx = sxf - sx0;
            const unsigned char* p00 = src + (sy0 * sw + sx0) * 3;
            const unsigned char* p01 = src + (sy0 * sw + sx1) * 3;
            const unsigned char* p10 = src + (sy1 * sw + sx0) * 3;
            const unsigned char* p11 = src + (sy1 * sw + sx1) * 3;
            unsigned char* d = dst + (y * INPUT_SIZE + x) * 3;
            for (int c = 0; c < 3; c++) {
                float v = p00[c] * (1 - wx) * (1 - wy) + p01[c] * wx * (1 - wy) +
                          p10[c] * (1 - wx) * wy + p11[c] * wx * wy;
                d[swap_rb ? 2 - c : c] = (unsigned char)(v + 0.5f);
            }
        }
    }
}

static inline unsigned char clamp_u8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

/* NV21 (Y 平面 + VU 交织) → BGR u8 全尺寸（BT.601 video range 定点） */
static void nv21_to_bgr(const unsigned char* nv21, int w, int h, unsigned char* bgr) {
    const unsigned char* yp = nv21;
    const unsigned char* vu = nv21 + w * h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int C = 298 * ((int)yp[y * w + x] - 16);
            const unsigned char* p = vu + (y / 2) * w + (x / 2) * 2; /* [V, U] */
            int E = (int)p[0] - 128, D = (int)p[1] - 128;
            unsigned char* d = bgr + (y * w + x) * 3;
            d[0] = clamp_u8((C + 516 * D + 128) >> 8);           /* B */
            d[1] = clamp_u8((C - 100 * D - 208 * E + 128) >> 8); /* G */
            d[2] = clamp_u8((C + 409 * E + 128) >> 8);           /* R */
        }
    }
}

/* NV21 → 640x640 RGB 一步融合(Y 定点双线性 Q8 + UV 最近邻)——全整数, 预算 x 映射表 */
static void nv21_resize_to_input(const unsigned char* nv21, int w, int h, unsigned char* dst) {
    const unsigned char* yp = nv21;
    const unsigned char* vu = nv21 + w * h;
    /* 预计算 x 方向映射(每帧一次, 640 项): sx0/sx1/wx(Q8) + UV 列偏移 */
    static int t_w = -1;
    static int sx0t[INPUT_SIZE], sx1t[INPUT_SIZE], wxt[INPUT_SIZE], uvxt[INPUT_SIZE];
    static int sy0t[INPUT_SIZE], sy1t[INPUT_SIZE], wyt[INPUT_SIZE];
    static int t_h = -1;
    if (t_w != w) {
        t_w = w;
        for (int x = 0; x < INPUT_SIZE; x++) {
            int fxq = ((2 * x + 1) * w * 128) / INPUT_SIZE - 128; /* Q8 源坐标 */
            if (fxq < 0) fxq = 0;
            sx0t[x] = fxq >> 8;
            sx1t[x] = sx0t[x] + 1 < w ? sx0t[x] + 1 : w - 1;
            wxt[x] = fxq & 255;
            uvxt[x] = (sx0t[x] / 2) * 2;
        }
    }
    if (t_h != h) {
        t_h = h;
        for (int y = 0; y < INPUT_SIZE; y++) {
            int fyq = ((2 * y + 1) * h * 128) / INPUT_SIZE - 128;
            if (fyq < 0) fyq = 0;
            sy0t[y] = fyq >> 8;
            sy1t[y] = sy0t[y] + 1 < h ? sy0t[y] + 1 : h - 1;
            wyt[y] = fyq & 255;
        }
    }
    for (int y = 0; y < INPUT_SIZE; y++) {
        const unsigned char* r0 = yp + sy0t[y] * w;
        const unsigned char* r1 = yp + sy1t[y] * w;
        const unsigned char* uvr = vu + (sy0t[y] / 2) * w;
        const int wy = wyt[y], iwy = 256 - wy;
        unsigned char* d = dst + (size_t)y * INPUT_SIZE * 3;
        for (int x = 0; x < INPUT_SIZE; x++, d += 3) {
            const int sx0 = sx0t[x], sx1 = sx1t[x], wx = wxt[x], iwx = 256 - wx;
            /* Y 双线性 Q16 → 整数 */
            int Y = (r0[sx0] * iwx + r0[sx1] * wx) * iwy + (r1[sx0] * iwx + r1[sx1] * wx) * wy;
            Y = (Y + 32768) >> 16;
            const unsigned char* uvp = uvr + uvxt[x]; /* [V,U] 最近邻 */
            int C = 298 * (Y - 16);
            int E = (int)uvp[0] - 128, D = (int)uvp[1] - 128;
            d[0] = clamp_u8((C + 409 * E + 128) >> 8);           /* R */
            d[1] = clamp_u8((C - 100 * D - 208 * E + 128) >> 8); /* G */
            d[2] = clamp_u8((C + 516 * D + 128) >> 8);           /* B */
        }
    }
}

/* ---------- 解码（对齐 decode_ref.py） ---------- */

static float iou(const det_t* a, const det_t* b) {
    float x1 = a->x1 > b->x1 ? a->x1 : b->x1, y1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float x2 = a->x2 < b->x2 ? a->x2 : b->x2, y2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float iw = x2 - x1 > 0 ? x2 - x1 : 0, ih = y2 - y1 > 0 ? y2 - y1 : 0;
    float inter = iw * ih;
    float ua = (a->x2 - a->x1) * (a->y2 - a->y1) + (b->x2 - b->x1) * (b->y2 - b->y1) - inter;
    return ua > 0 ? inter / ua : 0;
}

static int cmp_det(const void* pa, const void* pb) {
    float d = ((const det_t*)pb)->score - ((const det_t*)pa)->score;
    return d > 0 ? 1 : (d < 0 ? -1 : 0);
}

/* cls[3]/reg[3]: CHW float 输出；返回 NMS 后检出数 */
static int decode(float* const cls[3], float* const reg[3], int ow, int oh, float conf,
                  float nms_thr, det_t* out) {
    const float thr_logit = logf(conf / (1.0f - conf));
    const float sx = (float)ow / INPUT_SIZE, sy = (float)oh / INPUT_SIZE;
    det_t cand[MAX_DETS * 4];
    int nc = 0;

    for (int s = 0; s < 3; s++) {
        const int g = GRIDS[s], hw = g * g, stride = STRIDES[s];
        for (int i = 0; i < hw && nc < MAX_DETS * 4 - NUM_CLASS; i++) {
            /* 先用 logit 阈值筛（sigmoid 单调），过筛才算 DFL */
            int hit = 0;
            for (int c = 0; c < NUM_CLASS; c++)
                if (cls[s][c * hw + i] >= thr_logit) { hit = 1; break; }
            if (!hit) continue;

            /* DFL: 4 边 × 17 bin softmax 期望（每 anchor 只算一次） */
            float d[4];
            for (int k = 0; k < 4; k++) {
                const float* r = reg[s] + (k * (REG_MAX + 1)) * hw + i;
                float mx = r[0];
                for (int j = 1; j <= REG_MAX; j++)
                    if (r[j * hw] > mx) mx = r[j * hw];
                float sum = 0, acc = 0;
                for (int j = 0; j <= REG_MAX; j++) {
                    float e = expf(r[j * hw] - mx);
                    sum += e;
                    acc += j * e;
                }
                d[k] = acc / sum;
            }
            const float cx = ((i % g) + 0.5f) * stride, cy = ((i / g) + 0.5f) * stride;
            float x1 = (cx - d[0] * stride) * sx, y1 = (cy - d[1] * stride) * sy;
            float x2 = (cx + d[2] * stride) * sx, y2 = (cy + d[3] * stride) * sy;
            if (x1 < 0) x1 = 0;
            if (y1 < 0) y1 = 0;
            if (x2 > ow) x2 = ow;
            if (y2 > oh) y2 = oh;

            for (int c = 0; c < NUM_CLASS; c++) {
                float lg = cls[s][c * hw + i];
                if (lg < thr_logit) continue;
                det_t* t = &cand[nc++];
                t->x1 = x1; t->y1 = y1; t->x2 = x2; t->y2 = y2;
                t->score = 1.0f / (1.0f + expf(-lg));
                t->cls = c;
            }
        }
    }

    /* 逐类贪心 NMS */
    qsort(cand, nc, sizeof(det_t), cmp_det);
    int nk = 0;
    for (int i = 0; i < nc && nk < MAX_DETS; i++) {
        int keep = 1;
        for (int j = 0; j < nk; j++)
            if (out[j].cls == cand[i].cls && iou(&out[j], &cand[i]) > nms_thr) { keep = 0; break; }
        if (keep) out[nk++] = cand[i];
    }
    return nk;
}

/* ---------- 画框 ---------- */

static void draw_dets(awcv_mat_t* img, const det_t* dets, int n) {
    const int W = awcv_mat_cols(img);
    const int th = W > 1200 ? 4 : 2;
    const double fs = W > 1200 ? 0.9 : 0.55;
    for (int i = 0; i < n; i++) {
        const unsigned char* col = COLORS[dets[i].cls];
        awcv_scalar_t c = awcv_scalar_create(col[0], col[1], col[2], 0);
        awcv_point_t p1 = awcv_point_create((int)dets[i].x1, (int)dets[i].y1);
        awcv_point_t p2 = awcv_point_create((int)dets[i].x2, (int)dets[i].y2);
        awcv_rectangle_points(img, p1, p2, c, th);

        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", NAMES[dets[i].cls], dets[i].score);
        int base = 0;
        awcv_size_t ts = awcv_get_text_size(label, AWCV_C_FONT_HERSHEY_SIMPLEX, fs, th / 2 + 1, &base);
        int ty = (int)dets[i].y1 - 4;
        if (ty - ts.height < 0) ty = (int)dets[i].y1 + ts.height + 4; /* 顶部放不下移框内 */
        awcv_rect_t bg = awcv_rect_create((int)dets[i].x1, ty - ts.height - base / 2, ts.width + 4,
                                          ts.height + base);
        awcv_rectangle_rect(img, bg, c, AWCV_C_FILLED);
        awcv_scalar_t black = awcv_scalar_create(20, 20, 20, 0);
        awcv_put_text(img, label, awcv_point_create((int)dets[i].x1 + 2, ty),
                      AWCV_C_FONT_HERSHEY_SIMPLEX, fs, black, th / 2 + 1);
    }
}

static const char* base_name(const char* p) {
    const char* b = strrchr(p, '/');
    return b ? b + 1 : p;
}

int main(int argc, char** argv) {
    const char* model_dir = "model";
    const char* out_dir = "out";
    float conf = 0.45f, nms_thr = 0.50f;
    int draw = 1, nv21_w = 0, nv21_h = 0;
    int ot = -1; /* --ot 0..5 → LOWEST..MAXIMUM 带宽档位, -1=默认 */
    int first_input = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) conf = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) nms_thr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-draw")) draw = 0;
        else if (!strcmp(argv[i], "--ot") && i + 1 < argc) ot = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nv21") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &nv21_w, &nv21_h) != 2) {
                fprintf(stderr, "--nv21 格式: WxH (如 640x640)\n");
                return 1;
            }
        } else { first_input = i; break; }
    }
    if (!first_input) {
        fprintf(stderr,
                "用法: %s [-m model_dir] [-o out_dir] [-c conf] [-n nms] [--no-draw]\n"
                "          [--nv21 WxH] img1 [img2 ...]\n", argv[0]);
        return 1;
    }

    char param_path[512], model_path[512], jsonl_path[512];
    snprintf(param_path, sizeof(param_path), "%s/ppyoloe_s_640_ipu.param", model_dir);
    snprintf(model_path, sizeof(model_path), "%s/ppyoloe_s_640_ipu.bin", model_dir);
    snprintf(jsonl_path, sizeof(jsonl_path), "%s/results.jsonl", out_dir);

    /* ---- AWNN 初始化: 静态模式, 实例常驻 ---- */
    awnn_config_t config;
    memset(&config, 0, sizeof(config));
    config.precompiler_enable = true; /* ★ 静态模式: 43ms/帧, 内存恒定 */
    config.param_path = param_path;
    config.model_path = model_path;

    if (awnn_init() != 0) { fprintf(stderr, "awnn_init 失败\n"); return 1; }
    if (ot >= 0) {
        static const awnn_ot_type_t OTS[6] = {AWNN_C_OT_LOWEST, AWNN_C_OT_LOW, AWNN_C_OT_MEDIUM,
                                              AWNN_C_OT_HIGH, AWNN_C_OT_HIGHEST, AWNN_C_OT_MAXIMUM};
        int r = awnn_set_npu_ot(OTS[ot > 5 ? 5 : ot]);
        printf("set_npu_ot(%d) -> %d\n", ot, r);
    }
    awnn_instance_t* inst = awnn_instance_create(&config);
    if (!inst) { fprintf(stderr, "instance_create 失败(检查模型路径 %s)\n", model_dir); return 1; }

    unsigned char* input_buf = (unsigned char*)malloc(INPUT_SIZE * INPUT_SIZE * 3);
    awnn_tensor_desc_t in_t;
    memset(&in_t, 0, sizeof(in_t));
    in_t.layout = AWNN_C_LAYOUT_HWC;
    in_t.data_type = AWNN_C_DATA_TYPE_INT8;
    in_t.dims.w = INPUT_SIZE; in_t.dims.h = INPUT_SIZE; in_t.dims.c = 3;
    in_t.size = INPUT_SIZE * INPUT_SIZE * 3;
    in_t.data = input_buf;

    awnn_tensor_desc_t out_t[6];
    float* out_buf[6];
    for (int j = 0; j < 6; j++) {
        memset(&out_t[j], 0, sizeof(out_t[j]));
        out_t[j].layout = AWNN_C_LAYOUT_CHW;
        out_t[j].data_type = AWNN_C_DATA_TYPE_FP32;
        int c = (j % 2 == 0) ? NUM_CLASS : 4 * (REG_MAX + 1);
        int g = GRIDS[j / 2];
        out_buf[j] = (float*)malloc((size_t)c * g * g * sizeof(float)); /* 常驻复用 */
        out_t[j].data = out_buf[j];
        out_t[j].size = (uint32_t)c * g * g * sizeof(float);
    }

    const char* in_names[1] = {"image"};
    awnn_session_config_t sess;
    memset(&sess, 0, sizeof(sess));
    sess.type = AWNN_C_FORWARD_AUTO;
    sess.inputs_count = 1;
    sess.outputs_count = 6;
    sess.input_names = in_names;
    sess.output_names = OUT_NAMES;
    sess.input_tensors = &in_t;
    sess.output_tensors = out_t;

    double t0 = now_ms();
    if (awnn_instance_precompiler(inst, &sess) != 0) {
        fprintf(stderr, "precompiler 失败\n");
        awnn_instance_destroy(inst);
        return 1;
    }
    printf("precompile %.0fms (一次性)\n", now_ms() - t0);

    FILE* jf = fopen(jsonl_path, "w");
    if (!jf) { fprintf(stderr, "无法写 %s (out_dir 存在吗?)\n", jsonl_path); return 1; }

    unsigned char* nv21_bgr = NULL; /* NV21 全尺寸 BGR 缓冲(画框用), 按需分配 */
    det_t dets[MAX_DETS];
    int total = 0, ok_cnt = 0;
    double inf_sum = 0, inf_min = 1e9, inf_max = 0;

    for (int ai = first_input; ai < argc; ai++) {
        const char* path = argv[ai];
        if (!strcmp(path, "--nv21") && ai + 1 < argc) { /* 中途切换 NV21 尺寸 */
            sscanf(argv[++ai], "%dx%d", &nv21_w, &nv21_h);
            continue;
        }
        total++;
        double t_all = now_ms();
        double t_load0 = t_all;
        int ow = 0, oh = 0;
        awcv_mat_t* img = NULL;
        int is_nv21 = nv21_w > 0 && strstr(path, ".nv21") != NULL;

        if (is_nv21) {
            ow = nv21_w; oh = nv21_h;
            FILE* f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "[%s] 打不开\n", path); continue; }
            size_t need = (size_t)ow * oh * 3 / 2;
            unsigned char* raw = (unsigned char*)malloc(need);
            size_t got = fread(raw, 1, need, f);
            fclose(f);
            if (got != need) {
                fprintf(stderr, "[%s] 大小 %zu != 期望 %zu(%dx%d NV21)\n", path, got, need, ow, oh);
                free(raw);
                continue;
            }
            nv21_resize_to_input(raw, ow, oh, input_buf); /* 融合: NV21→640²RGB 一步 */
            if (draw) { /* 仅画图才需要全幅 BGR */
                nv21_bgr = (unsigned char*)realloc(nv21_bgr, (size_t)ow * oh * 3);
                nv21_to_bgr(raw, ow, oh, nv21_bgr);
                img = awcv_mat_create_with_data(oh, ow, CV_8UC3, nv21_bgr);
            }
            free(raw);
        } else {
            img = awcv_imread(path, AWCV_C_LOAD_IMAGE_COLOR); /* BGR */
            if (!img || awcv_mat_empty(img)) {
                fprintf(stderr, "[%s] 解码失败\n", path);
                continue;
            }
            ow = awcv_mat_cols(img);
            oh = awcv_mat_rows(img);
            resize_to_input(awcv_mat_data(img), ow, oh, 1, input_buf); /* BGR→RGB */
        }

        double load_ms = now_ms() - t_load0;

        /* ---- 推理 ---- */
        if (awnn_instance_set_in_tensors(inst, &sess) != 0) {
            fprintf(stderr, "[%s] set_in_tensors 失败\n", path);
            if (img) awcv_mat_destroy(img);
            continue;
        }
        double t_inf = now_ms();
        if (awnn_instance_inference(inst, &sess) != 0) {
            fprintf(stderr, "[%s] inference 失败 —— 立即停止(切勿重试, 会挂驱动)\n", path);
            break; /* ★熔断: NPU 出错绝不硬砸 */
        }
        double inf_ms = now_ms() - t_inf;
        if (awnn_instance_get_out_tensors(inst, &sess) != 0) {
            fprintf(stderr, "[%s] get_out_tensors 失败\n", path);
            if (img) awcv_mat_destroy(img);
            continue;
        }

        /* ---- 解码 ---- */
        double t_dec = now_ms();
        float* cls[3] = {out_buf[0], out_buf[2], out_buf[4]};
        float* reg[3] = {out_buf[1], out_buf[3], out_buf[5]};
        int n = decode(cls, reg, ow, oh, conf, nms_thr, dets);
        double dec_ms = now_ms() - t_dec;

        /* ---- JSONL ---- */
        fprintf(jf, "{\"image\":\"%s\",\"w\":%d,\"h\":%d,\"infer_ms\":%.1f,\"dets\":[", base_name(path), ow, oh, inf_ms);
        for (int i = 0; i < n; i++)
            fprintf(jf, "%s{\"label\":\"%s\",\"score\":%.3f,\"box\":[%.1f,%.1f,%.1f,%.1f]}",
                    i ? "," : "", NAMES[dets[i].cls], dets[i].score, dets[i].x1, dets[i].y1,
                    dets[i].x2, dets[i].y2);
        fprintf(jf, "]}\n");
        fflush(jf);

        /* ---- 标框图 ---- */
        if (draw && img) {
            draw_dets(img, dets, n);
            char op[512], nb[256];
            snprintf(nb, sizeof(nb), "%s", base_name(path));
            char* dot = strrchr(nb, '.');
            if (dot) *dot = 0;
            snprintf(op, sizeof(op), "%s/%s_det.jpg", out_dir, nb);
            int prm[2] = {AWCV_C_IMWRITE_JPEG_QUALITY, 90};
            if (!awcv_imwrite(op, img, prm, 2)) fprintf(stderr, "[%s] imwrite 失败\n", op);
        }
        if (img) awcv_mat_destroy(img);

        ok_cnt++;
        inf_sum += inf_ms;
        if (inf_ms < inf_min) inf_min = inf_ms;
        if (inf_ms > inf_max) inf_max = inf_ms;
        printf("[%d] %s  %dx%d  load=%.1f infer=%.1f dec=%.1f total=%.1fms  dets=%d", ok_cnt,
               base_name(path), ow, oh, load_ms, inf_ms, dec_ms, now_ms() - t_all, n);
        for (int i = 0; i < n && i < 4; i++)
            printf("  %s:%.2f", NAMES[dets[i].cls], dets[i].score);
        printf("\n");
        fflush(stdout);
    }

    fclose(jf);
    printf("---- %d/%d 张成功  infer min/avg/max = %.1f/%.1f/%.1f ms ----\n", ok_cnt, total,
           ok_cnt ? inf_min : 0, ok_cnt ? inf_sum / ok_cnt : 0, ok_cnt ? inf_max : 0);
    awnn_eval_npu_memory();
    {   /* 进程内存自报 (峰值 VmHWM / 当前 VmRSS) */
        FILE* st = fopen("/proc/self/status", "r");
        if (st) {
            char ln[128];
            while (fgets(ln, sizeof(ln), st))
                if (!strncmp(ln, "VmHWM", 5) || !strncmp(ln, "VmRSS", 5)) printf("%s", ln);
            fclose(st);
        }
    }

    awnn_instance_destroy(inst);
    awnn_deinit();
    free(input_buf);
    free(nv21_bgr);
    for (int j = 0; j < 6; j++) free(out_buf[j]);
    return 0;
}
