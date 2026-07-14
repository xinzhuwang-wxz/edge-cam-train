/*
 * Copyright (c) 2019-2025 Allwinner Technology Co., Ltd. ALL rights reserved.
 *
 * Allwinner is a trademark of Allwinner Technology Co.,Ltd., registered in
 * the people's Republic of China and other countries.
 * All Allwinner Technology Co.,Ltd. trademarks are used with permission.
 *
 * DISCLAIMER
 * THIRD PARTY LICENCES MAY BE REQUIRED TO IMPLEMENT THE SOLUTION/PRODUCT.
 * IF YOU NEED TO INTEGRATE THIRD PARTY'S TECHNOLOGY (SONY, DTS, DOLBY, AVS OR MPEGLA, ETC.)
 * IN ALLWINNERS'SDK OR PRODUCTS, YOU SHALL BE SOLELY RESPONSIBLE TO OBTAIN
 * ALL APPROPRIATELY REQUIRED THIRD PARTY LICENCES.
 * ALLWINNER SHALL HAVE NO WARRANTY, INDEMNITY OR OTHER OBLIGATIONS WITH RESPECT TO MATTERS
 * COVERED UNDER ANY REQUIRED THIRD PARTY LICENSE.
 * YOU ARE SOLELY RESPONSIBLE FOR YOUR USAGE OF THIRD PARTY'S TECHNOLOGY.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY ALLWINNER"AS IS" AND TO THE MAXIMUM EXTENT
 * PERMITTED BY LAW, ALLWINNER EXPRESSLY DISCLAIMS ALL WARRANTIES OF ANY KIND,
 * WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING WITHOUT LIMITATION REGARDING
 * THE TITLE, NON-INFRINGEMENT, ACCURACY, CONDITION, COMPLETENESS, PERFORMANCE
 * OR MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL ALLWINNER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS, OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>

#include "AWNN_interface.h"
#include "aw_image.h"
#include "aw_fmt_cvt.h"

double getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int loadFromBin(const char* binPath, int size, void* buffer)
{
    if (buffer == NULL)
    {
        fprintf(stderr, "loadFromBin error: buffer is NULL.\n");
        return -1;
    }
    FILE* fp = fopen(binPath, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "fopen %s failed\n", binPath);
        return -1;
    }
    int nread = fread(buffer, 1, size, fp);
    if (nread != size)
    {
        fprintf(stderr, "fread bin failed, size %d dismatch.\n", nread);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int compareResult(const char* trueResultPath, const signed char* inferenceResult, int size)
{
    signed char* trueResult = (signed char*)malloc(size);
    int ret = loadFromBin(trueResultPath, size, trueResult);
    if (ret != 0)
    {
        fprintf(stderr, "load result %s error.\n", trueResultPath);
        return ret;
    }

    int countSuccess = 0, countFail = 0;
    float mse = 0.0, snr = 0.0, cos_sim = 0.0;
    int sum_dot = 0, sum_diff = 0, sum_true = 0, sum_infer = 0;

    for (int i = 0; i < size; i++)
    {
        int tmp = trueResult[i] - inferenceResult[i];
        int diff = tmp >= 0 ? tmp : (-tmp);
        if (diff == 0) countSuccess++;
        else countFail++;
        sum_diff += diff * diff;
        sum_true += trueResult[i] * trueResult[i];
        sum_infer += inferenceResult[i] * inferenceResult[i];
        sum_dot += (trueResult[i] * inferenceResult[i]);
    }

    if (countFail == 0)
    {
        fprintf(stderr, "%s: test success ^_^ ^_^ ^_^\n", trueResultPath);
    }
    else
    {
        fprintf(stderr, "%s, test fail T_T T_T T_T\n", trueResultPath);
        mse = ((float)sum_diff) / size;
        snr = ((float)sum_diff) / (sum_true + 1e-7);
        cos_sim = ((float)sum_dot) / (sqrt(sum_true) * sqrt(sum_infer) + 1e-7);
        fprintf(stderr, "output similarity: mse = %f, snr = %f, cos = %f\n", mse, snr, cos_sim);
    }
    fprintf(stderr, "count success num: %d, count fail num: %d\n", countSuccess, countFail);

    free(trueResult);
    return countFail;
}

static int compareResultFP(const char* trueResultPath, float* inferenceResult, int size, float epsilon)
{
    float* trueResult = (float*)malloc(size * sizeof(float));
    int ret = loadFromBin(trueResultPath, size * sizeof(float), trueResult);
    if (ret != 0)
    {
        fprintf(stderr, "load result %s error.\n", trueResultPath);
        return ret;
    }

    int countSuccess = 0, countFail = 0;
    float mse = 0.0, snr = 0.0, cos_sim = 0.0;
    float sum_dot = 0.0, sum_diff = 0.0, sum_true = 0.0, sum_infer = 0.0;

    for (int i = 0; i < size; i++)
    {
        float tmp = trueResult[i] - inferenceResult[i];
        float diff = tmp >= 0 ? tmp : (-tmp);
        if (diff < epsilon) countSuccess++;
        else countFail++;
        sum_diff += diff * diff;
        sum_true += trueResult[i] * trueResult[i];
        sum_infer += inferenceResult[i] * inferenceResult[i];
        sum_dot += (trueResult[i] * inferenceResult[i]);
    }

    if (countFail == 0)
    {
        fprintf(stderr, "%s: test success ^_^ ^_^ ^_^\n", trueResultPath);
    }
    else
    {
        fprintf(stderr, "%s, test fail T_T T_T T_T\n", trueResultPath);
        mse = sum_diff / size;
        snr = sum_diff / (sum_true + 1e-7);
        cos_sim = sum_dot / (sqrt(sum_true) * sqrt(sum_infer) + 1e-7);
        fprintf(stderr, "output similarity: mse = %f, snr = %f, cos = %f\n", mse, snr, cos_sim);
    }
    fprintf(stderr, "count success num: %d, count fail num: %d\n", countSuccess, countFail);

    free(trueResult);
    return countFail;
}

/* Simple C config parser */
typedef struct {
    char key[256];
    char value[1024];
} ConfigEntry;

typedef struct {
    ConfigEntry entries[128];
    int count;
} SimpleConfig;

static void trimString(char* s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    char* end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) *end = '\0', end--;
}

static int parseConfig(const char* filePath, SimpleConfig* cfg)
{
    FILE* fp = fopen(filePath, "r");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", filePath);
        return -1;
    }

    cfg->count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp) && cfg->count < 128)
    {
        char* comment = strchr(line, '#');
        if (comment) *comment = '\0';

        char* delim = strchr(line, '=');
        if (!delim) continue;

        *delim = '\0';
        char* key = line;
        char* value = delim + 1;

        trimString(key);
        trimString(value);

        if (strlen(key) > 0)
        {
            strncpy(cfg->entries[cfg->count].key, key, 255);
            strncpy(cfg->entries[cfg->count].value, value, 1023);
            cfg->count++;
        }
    }
    fclose(fp);
    return 0;
}

static const char* configReadStr(SimpleConfig* cfg, const char* key, const char* defaultVal)
{
    for (int i = 0; i < cfg->count; i++)
    {
        if (strcmp(cfg->entries[i].key, key) == 0)
            return cfg->entries[i].value;
    }
    return defaultVal;
}

static int configReadInt(SimpleConfig* cfg, const char* key, int defaultVal)
{
    const char* val = configReadStr(cfg, key, NULL);
    if (!val) return defaultVal;
    return atoi(val);
}

static int splitString(const char* input, char delimiter, char** outputs, int maxOutputs)
{
    int count = 0;
    const char* start = input;
    while (*start && count < maxOutputs)
    {
        const char* end = strchr(start, delimiter);
        if (!end) end = start + strlen(start);

        int len = end - start;
        outputs[count] = (char*)malloc(len + 1);
        strncpy(outputs[count], start, len);
        outputs[count][len] = '\0';

        char* s = outputs[count];
        trimString(s);

        count++;
        if (*end == delimiter) start = end + 1;
        else break;
    }
    return count;
}

static int splitString2Int(const char* input, char delimiter, int* outputs, int maxOutputs)
{
    char** strs = (char**)malloc(maxOutputs * sizeof(char*));
    int count = splitString(input, delimiter, strs, maxOutputs);
    for (int i = 0; i < count; i++)
    {
        outputs[i] = atoi(strs[i]);
        free(strs[i]);
    }
    free(strs);
    return count;
}

typedef struct {
    char** input_paths;
    char** output_paths;
    char** input_blob_names;
    char** output_blob_names;
    int* inputs_w;
    int* inputs_h;
    int* inputs_c;
    char** input_data_type;
    char** output_data_type;
    char model_path[512];
    char param_path[512];
    int use_static_mode;
    int use_awnn_profiler;
    int is_compare_result;
    int dump_output_result;
    char net_name[256];
    int loop_count;
    int inputs_count;
    int outputs_count;
    char batch_list[512];   /* 批量模式: 每行一个输入 .bin 路径 */
    char out_dir[512];      /* 批量模式: 输出目录 */
} ConfigInfo;

static int checkConfigInfo(const ConfigInfo* configInfo)
{
    if (configInfo->inputs_count <= 0 || configInfo->outputs_count <= 0)
    {
        fprintf(stderr, "ConfigInfo error: invalid inputs/outputs count\n");
        return -1;
    }
    if (strlen(configInfo->model_path) == 0 || strlen(configInfo->param_path) == 0)
    {
        fprintf(stderr, "ConfigInfo error: model_path or param_path is empty\n");
        return -1;
    }
    if (strlen(configInfo->net_name) == 0)
    {
        fprintf(stderr, "ConfigInfo error: net_name is empty\n");
        return -1;
    }
    if (configInfo->loop_count <= 0)
    {
        fprintf(stderr, "ConfigInfo error: loop_count invalid\n");
        return -1;
    }
    return 0;
}

static void freeConfigInfo(ConfigInfo* configInfo)
{
    for (int i = 0; i < configInfo->inputs_count; i++)
    {
        free(configInfo->input_paths[i]);
        free(configInfo->input_blob_names[i]);
        free(configInfo->input_data_type[i]);
    }
    for (int i = 0; i < configInfo->outputs_count; i++)
    {
        free(configInfo->output_paths[i]);
        free(configInfo->output_blob_names[i]);
        free(configInfo->output_data_type[i]);
    }
    free(configInfo->input_paths);
    free(configInfo->output_paths);
    free(configInfo->input_blob_names);
    free(configInfo->output_blob_names);
    free(configInfo->inputs_w);
    free(configInfo->inputs_h);
    free(configInfo->inputs_c);
    free(configInfo->input_data_type);
    free(configInfo->output_data_type);
}

static int initConfigInfo(SimpleConfig* cfg, ConfigInfo* configInfo)
{
    const char* input_paths_str = configReadStr(cfg, "input_paths", "");
    const char* output_paths_str = configReadStr(cfg, "output_paths", "");
    const char* input_blob_names_str = configReadStr(cfg, "input_blob_names", "");
    const char* output_blob_names_str = configReadStr(cfg, "output_blob_names", "");
    const char* inputs_w_str = configReadStr(cfg, "inputs_w", "");
    const char* inputs_h_str = configReadStr(cfg, "inputs_h", "");
    const char* inputs_c_str = configReadStr(cfg, "inputs_c", "");
    const char* input_data_type_str = configReadStr(cfg, "input_data_type", "");
    const char* output_data_type_str = configReadStr(cfg, "output_data_type", "");

    strncpy(configInfo->model_path, configReadStr(cfg, "model_path", ""), 511);
    strncpy(configInfo->param_path, configReadStr(cfg, "param_path", ""), 511);
    configInfo->use_static_mode = configReadInt(cfg, "use_static_mode", 0);
    configInfo->use_awnn_profiler = configReadInt(cfg, "use_awnn_profiler", 0);
    configInfo->is_compare_result = configReadInt(cfg, "is_compare_result", 0);
    configInfo->dump_output_result = configReadInt(cfg, "dump_output_result", 0);
    strncpy(configInfo->net_name, configReadStr(cfg, "net_name", ""), 255);
    configInfo->loop_count = configReadInt(cfg, "loop_count", 1);
    strncpy(configInfo->batch_list, configReadStr(cfg, "batch_list", ""), 511);
    strncpy(configInfo->out_dir, configReadStr(cfg, "out_dir", "."), 511);

    configInfo->inputs_count = 8;
    configInfo->outputs_count = 8;

    configInfo->input_paths = (char**)malloc(configInfo->inputs_count * sizeof(char*));
    configInfo->input_blob_names = (char**)malloc(configInfo->inputs_count * sizeof(char*));
    configInfo->input_data_type = (char**)malloc(configInfo->inputs_count * sizeof(char*));
    configInfo->inputs_w = (int*)malloc(configInfo->inputs_count * sizeof(int));
    configInfo->inputs_h = (int*)malloc(configInfo->inputs_count * sizeof(int));
    configInfo->inputs_c = (int*)malloc(configInfo->inputs_count * sizeof(int));

    configInfo->inputs_count = splitString(input_paths_str, ',', configInfo->input_paths, 8);
    int blobCount = splitString(input_blob_names_str, ',', configInfo->input_blob_names, 8);
    int dtypeCount = splitString(input_data_type_str, ',', configInfo->input_data_type, 8);
    splitString2Int(inputs_w_str, ',', configInfo->inputs_w, 8);
    splitString2Int(inputs_h_str, ',', configInfo->inputs_h, 8);
    splitString2Int(inputs_c_str, ',', configInfo->inputs_c, 8);

    configInfo->output_paths = (char**)malloc(8 * sizeof(char*));
    configInfo->output_blob_names = (char**)malloc(8 * sizeof(char*));
    configInfo->output_data_type = (char**)malloc(8 * sizeof(char*));

    configInfo->outputs_count = splitString(output_paths_str, ',', configInfo->output_paths, 8);
    splitString(output_blob_names_str, ',', configInfo->output_blob_names, 8);
    splitString(output_data_type_str, ',', configInfo->output_data_type, 8);

    return checkConfigInfo(configInfo);
}

static int awnnVerify(const ConfigInfo* configInfo)
{
    /* set awnn_config_t */
    awnn_config_t config;
    memset(&config, 0, sizeof(config));
    config.param_invisible = false;
    config.profiler_enable = configInfo->use_awnn_profiler ? true : false;
    config.precompiler_enable = configInfo->use_static_mode ? true : false;
    config.param_path = configInfo->param_path;
    config.model_path = configInfo->model_path;

    /* create awnn instance */
    awnn_instance_t* instance = awnn_instance_create(&config);
    if (!instance)
    {
        fprintf(stderr, "awnn_instance_create error.\n");
        return -1;
    }
    printf("finish creating awnn instance.\n");

    /* allocate input buffers */
    void** inputBuffers = (void**)malloc(configInfo->inputs_count * sizeof(void*));
    for (int i = 0; i < configInfo->inputs_count; i++)
    {
        int inputSize = configInfo->inputs_w[i] * configInfo->inputs_h[i] * configInfo->inputs_c[i];
        if (strcmp(configInfo->input_data_type[i], "DATA_TYPE_FP32") == 0)
            inputSize *= sizeof(float);
        inputBuffers[i] = malloc(inputSize);
    }
    printf("finish setting input buffer.\n");

    /* set input tensors */
    awnn_tensor_desc_t* inputTensors = (awnn_tensor_desc_t*)malloc(configInfo->inputs_count * sizeof(awnn_tensor_desc_t));
    for (int i = 0; i < configInfo->inputs_count; i++)
    {
        memset(&inputTensors[i], 0, sizeof(awnn_tensor_desc_t));
        inputTensors[i].dims.w = configInfo->inputs_w[i];
        inputTensors[i].dims.h = configInfo->inputs_h[i];
        inputTensors[i].dims.c = configInfo->inputs_c[i];
        inputTensors[i].size = configInfo->inputs_w[i] * configInfo->inputs_h[i] * configInfo->inputs_c[i];
        inputTensors[i].data = inputBuffers[i];
        if (strcmp(configInfo->input_data_type[i], "DATA_TYPE_FP32") == 0)
        {
            inputTensors[i].layout = AWNN_C_LAYOUT_CHW;
            inputTensors[i].data_type = AWNN_C_DATA_TYPE_FP32;
        }
        else
        {
            inputTensors[i].layout = AWNN_C_LAYOUT_HWC;
            inputTensors[i].data_type = AWNN_C_DATA_TYPE_INT8;
        }
    }
    printf("finish setting input information.\n");

    /* set output tensors */
    awnn_tensor_desc_t* outputTensors = (awnn_tensor_desc_t*)malloc(configInfo->outputs_count * sizeof(awnn_tensor_desc_t));
    for (int j = 0; j < configInfo->outputs_count; j++)
    {
        memset(&outputTensors[j], 0, sizeof(awnn_tensor_desc_t));
        if (strcmp(configInfo->output_data_type[j], "DATA_TYPE_FP32") == 0)
        {
            outputTensors[j].layout = AWNN_C_LAYOUT_CHW;
            outputTensors[j].data_type = AWNN_C_DATA_TYPE_FP32;
        }
        else if (strcmp(configInfo->output_data_type[j], "DATA_TYPE_FP32_CDHW") == 0)
        {
            outputTensors[j].layout = AWNN_C_LAYOUT_CDHW;
            outputTensors[j].data_type = AWNN_C_DATA_TYPE_FP32;
        }
        else
        {
            outputTensors[j].layout = AWNN_C_LAYOUT_HWC;
            outputTensors[j].data_type = AWNN_C_DATA_TYPE_INT8;
        }
    }
    printf("finish setting output information.\n");

    /* set session config */
    awnn_session_config_t sessConfig;
    memset(&sessConfig, 0, sizeof(sessConfig));
    sessConfig.type = AWNN_C_FORWARD_AUTO;
    sessConfig.inputs_count = configInfo->inputs_count;
    sessConfig.outputs_count = configInfo->outputs_count;
    sessConfig.input_names = (const char**)configInfo->input_blob_names;
    sessConfig.output_names = (const char**)configInfo->output_blob_names;
    sessConfig.input_tensors = inputTensors;
    sessConfig.output_tensors = outputTensors;

    if (config.precompiler_enable)
    {
        int compileFlag = awnn_instance_precompiler(instance, &sessConfig);
        if (compileFlag != 0)
        {
            fprintf(stderr, "precompiler error.\n");
            awnn_instance_destroy(instance);
            for (int i = 0; i < configInfo->inputs_count; i++) free(inputBuffers[i]);
            free(inputBuffers);
            free(inputTensors);
            free(outputTensors);
            return -1;
        }
        printf("finish precompiler.\n");
    }

    double timeMin = DBL_MAX;
    double timeMax = -DBL_MAX;
    double timeAvg = 0;

    /* ★ 批量模式: 从 batch_list 读入 N 个输入路径, 一个进程内跑完
       (反复 exec awnn_verify 会让 NPU carveout 不释放 -> DMA_HEAP_IOCTL_ALLOC failed -> 驱动卡死) */
    char** batchPaths = NULL;
    int batchCount = 0;
    if (strlen(configInfo->batch_list) > 0)
    {
        FILE* lf = fopen(configInfo->batch_list, "r");
        if (!lf) { fprintf(stderr, "open batch_list error: %s\n", configInfo->batch_list); return -1; }
        batchPaths = (char**)malloc(4096 * sizeof(char*));
        char line[512];
        while (batchCount < 4096 && fgets(line, sizeof(line), lf))
        {
            int L = strlen(line);
            while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r' || line[L-1] == ' ')) line[--L] = 0;
            if (L == 0) continue;
            batchPaths[batchCount] = strdup(line);
            batchCount++;
        }
        fclose(lf);
        printf("batch mode: %d inputs from %s\n", batchCount, configInfo->batch_list);
    }
    int nRuns = batchCount > 0 ? batchCount : configInfo->loop_count;

    for (int i = 0; i < nRuns; i++)
    {
        const char* curIn = batchCount > 0 ? batchPaths[i] : configInfo->input_paths[0];

        /* load input bin */
        for (int j = 0; j < configInfo->inputs_count; j++)
        {
            int inputSize = configInfo->inputs_w[j] * configInfo->inputs_h[j] * configInfo->inputs_c[j];
            if (strcmp(configInfo->input_data_type[j], "DATA_TYPE_FP32") == 0)
                inputSize *= sizeof(float);
            const char* p = (batchCount > 0 && j == 0) ? curIn : configInfo->input_paths[j];
            int ret = loadFromBin(p, inputSize, inputBuffers[j]);
            if (ret != 0)
            {
                fprintf(stderr, "load input bin error: %s\n", p);
                break;
            }
        }

        /* set input tensors */
        awnn_instance_set_in_tensors(instance, &sessConfig);

        double start = getCurrentTime();

        int inferenceFlag = awnn_instance_inference(instance, &sessConfig);

        double end = getCurrentTime();
        double time = end - start;
        timeMin = fmin(timeMin, time);
        timeMax = fmax(timeMax, time);
        timeAvg += time;

        if (inferenceFlag != 0)
        {
            fprintf(stderr, "inference error at %s\n", curIn);
            break;
        }
        printf("[%d/%d] %s  npu_ms=%.2f\n", i + 1, nRuns, curIn, time);
        fflush(stdout);

        /* get output tensors */
        for (int j = 0; j < configInfo->outputs_count; j++)
        {
            awnn_tensor_dims_t outDims;
            awnn_instance_get_out_tensor_dims_by_name(instance, configInfo->output_blob_names[j], &outDims);

            int outputSize = 0;
            if (outDims.d > 0)
            {
                outputSize = outDims.w * outDims.h * outDims.d * outDims.c;
                if (i == 0) printf("outputNames[%d] = %s, [w,h,d,c] = [%d,%d,%d,%d], size = %d\n",
                    j, configInfo->output_blob_names[j], outDims.w, outDims.h, outDims.d, outDims.c, outputSize);
            }
            else
            {
                outputSize = outDims.w * outDims.h * outDims.c;
                if (i == 0) printf("outputNames[%d] = %s, [w,h,c] = [%d,%d,%d], size = %d\n",
                    j, configInfo->output_blob_names[j], outDims.w, outDims.h, outDims.c, outputSize);
            }

            if (outputSize > 0)
            {
                if (strcmp(configInfo->output_data_type[j], "DATA_TYPE_INT8") != 0)
                    outputSize *= sizeof(float);
                outputTensors[j].data = malloc(outputSize);
            }
            else
            {
                fprintf(stderr, "outputSize error.\n");
                break;
            }
        }

        /* get output tensors */
        awnn_instance_get_out_tensors(instance, &sessConfig);

        if (configInfo->dump_output_result)
        {
            for (int m = 0; m < configInfo->outputs_count; m++)
            {
                char outPath[512];
                /* tag = 输入文件名去掉目录和 .bin */
                char tag[128] = "out";
                if (batchCount > 0)
                {
                    const char* b = strrchr(curIn, '/');
                    b = b ? b + 1 : curIn;
                    snprintf(tag, sizeof(tag), "%s", b);
                    char* d = strrchr(tag, '.');
                    if (d) *d = 0;
                }
                const char* sfx = (outputTensors[m].data_type == AWNN_C_DATA_TYPE_INT8) ? "hwc_int8"
                                : (outputTensors[m].layout == AWNN_C_LAYOUT_CDHW) ? "cdhw_fp32" : "chw_fp32";
                if (batchCount > 0)
                    snprintf(outPath, sizeof(outPath), "%s/%s__%s_awnn_%s.bin",
                             configInfo->out_dir, tag, configInfo->output_blob_names[m], sfx);
                else
                    snprintf(outPath, sizeof(outPath), "%s_awnn_%s.bin", configInfo->output_blob_names[m], sfx);
                FILE* mf = fopen(outPath, "wb");
                size_t elemSize = (outputTensors[m].data_type == AWNN_C_DATA_TYPE_INT8) ? 1 : 4;
                fwrite(outputTensors[m].data, 1, (size_t)outputTensors[m].size * elemSize, mf);
                fclose(mf);
            }
        }

        if (configInfo->is_compare_result)
        {
            for (int m = 0; m < configInfo->outputs_count; m++)
            {
                if (outputTensors[m].data_type == AWNN_C_DATA_TYPE_INT8)
                {
                    compareResult(configInfo->output_paths[m],
                        (signed char*)outputTensors[m].data, outputTensors[m].size);
                }
                else
                {
                    compareResultFP(configInfo->output_paths[m],
                        (float*)outputTensors[m].data, outputTensors[m].size, 0.001f);
                }
            }
        }

        for (int j = 0; j < configInfo->outputs_count; j++)
        {
            if (outputTensors[j].data)
            {
                free(outputTensors[j].data);
                outputTensors[j].data = NULL;
            }
        }
    }

    fprintf(stderr, "%s:  min = %7.2f  max = %7.2f  avg = %7.2f  (n=%d)\n",
        configInfo->net_name, timeMin, timeMax, timeAvg / (nRuns > 0 ? nRuns : 1), nRuns);
    for (int i = 0; i < batchCount; i++) free(batchPaths[i]);
    free(batchPaths);

    awnn_eval_npu_memory();
    awnn_instance_destroy(instance);

    for (int i = 0; i < configInfo->inputs_count; i++)
        free(inputBuffers[i]);
    free(inputBuffers);
    free(inputTensors);
    free(outputTensors);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [configPath]\n", argv[0]);
        fprintf(stderr, "example: %s ./config.txt\n", argv[0]);
        return -1;
    }

    awnn_print_version();

    int ret1 = awnn_init();
    if (ret1 == -1)
    {
        fprintf(stderr, "awnn_init error.\n");
        return -1;
    }

    SimpleConfig cfg;
    int ret = parseConfig(argv[1], &cfg);
    if (ret != 0)
    {
        fprintf(stderr, "parseConfig error.\n");
        awnn_deinit();
        return -1;
    }

    ConfigInfo configInfo;
    memset(&configInfo, 0, sizeof(configInfo));
    ret = initConfigInfo(&cfg, &configInfo);
    if (ret != 0)
    {
        fprintf(stderr, "initConfigInfo error.\n");
        awnn_deinit();
        return -1;
    }

    int flag = awnnVerify(&configInfo);
    if (flag != 0)
        fprintf(stderr, "AWNN Test Fail\n");
    else
        fprintf(stderr, "AWNN Test Pass\n");

    freeConfigInfo(&configInfo);

    int ret2 = awnn_deinit();
    if (ret2 == -1)
    {
        fprintf(stderr, "awnn_deinit error.\n");
        return -1;
    }

    return 0;
}