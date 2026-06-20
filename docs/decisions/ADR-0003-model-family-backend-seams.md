# ADR-0003：多模型族 = 按阶段切 seam,挂共享脊柱

- 状态：Accepted
- 日期：2026-06-20
- 相关：`docs/engineering.md` §2/§3（TrainerBackend/PackagerBackend 计划)、[[ADR-0001]]、[[ADR-0002]]、架构审查（improve-codebase-architecture：模型族扩展性）

## 背景

框架要容纳**两个对等的模型族**——检测(粗分类+框)与细分类——各自走全流程 `train→export→eval→publish→deploy`,并支持把二者**合成级联**。后续还要加 **YOLO 系**等新检测器。

审查发现现状不对称:**分类是 in-process Lightning 深模块**(`Classifier`/`ClassifyDataModule`/`export_onnx`,Hydra multirun);**检测是 subprocess 黑盒**(`run_nanodet` 调 NanoDet fork)。下游评估硬编分类:`run_full_eval(model: Classifier)`、`EnvelopeReport.top1/top5`、`promotion` 默认 `task="classify"` → **检测评估是孤立轨道(detect_ablation.csv),无法 gate/promote/OTA**。加新检测器 = 复制 `run_nanodet` + 改所有 caller 分支(浅、copy-paste)。

但脊柱已族无关:`PackagerBackend.pack()`(只要 FP32 ONNX)、`registry.store.list(task)`、`ModelCard.task`、`ChannelPolicy`。

## 决策

1. **按阶段切 seam,不做"每族一个全生命周期大对象"**。阶段的族耦合度不同 → seam 切在族相关处,族无关处共用:
   - **`TrainerBackend` Protocol**(train+export):`train(cfg) -> TrainedRef`(不透明句柄:分类=内存模型,subprocess=checkpoint 路径)+ `export_fp32_onnx(ref, out, input_size) -> Path`。每族一个 adapter(`ClassifyBackend`/`NanodetBackend`/未来 `YoloBackend` ~60 行),工厂按 `cfg.backend` 派发。
   - **`Evaluator` per 族 → 统一 `EvalReport`**:`EvalReport.levels[].metrics: dict[str,float]` + `primary` 指标名。分类级带 `top1/top5`,检测级带 `map_50/map_5095/bird_recall`。**不分裂成两个 report 子类**——dict 化指标,`metrics_from_report` 直接摊(ModelCard.metrics 本就是 dict)。
   - **脊柱族无关、两族共用**:`EvalReport → ModelCard(task) → registry → PackagerBackend → ChannelPolicy/OTA` 不重新发明。
2. **gate 泛化为"命名指标阈值集"**:`{int8_drop: 0.05}`(分类)/`{map_50_min: 0.6}`(检测),按 `primary` 指标判定。
3. **FP32 ONNX 契约收一处**(`onnx_artifact`):`export_fp32_onnx` 内统一过 `check_onnx_contract`(FP32+静态 shape,上 ACUITY 硬契约,总跑)+ 有 torch 模型时 `check_onnx_matches_torch`(数值对齐)。替代现有两套混淆命名的 `verify_onnx`/`verify_onnx_loadable`。
4. **级联是脊柱之上的组合层,不属任何单族**:`cascade/CascadePipeline(detector_onnx, classifier_onnx, crop_policy, gating_policy)` 暴露 `infer`(贯穿置信门控/层级回退)+ `evaluate`(检出率/级联 top-1/量化组合),复用 `data/crop.py`。
5. **数据脊柱:共享 Provenance 基 + 每族 manifest**。抽 `Provenance/DatasetSpec` 基(`name/version/splits` + 逐样本 `source/license/taxon_key`),两族 manifest 继承:分类用现有 `DatasetManifest`,检测**新增 `DetectionManifest`**(包 COCO images/annotations(bbox)+ 同款溯源,替代喂裸 COCO labels.json)。`TrainerBackend` 经各族 DataModule 读各族 manifest。**provenance 一路流到 ModelCard**(许可红线/署名/taxon 版本全链路可追溯,§4 海外发行从严)。检测 `bird` 粗类与分类 species 头经 [[ADR-0002]] eBird 规范键对接 → 级联在数据层即接得上。

6. **三个正交扩展点,各一个 adapter(成熟框架通法,对齐 MMDetection BaseDataset / Detectron2 DatasetCatalog）**:
   - **新数据源(同族)** = 一个 **ingest adapter**(`源 → Manifest`);manifest 即统一接口,下游 loader/训练/评估全复用。
   - **新模型族** = 一个 `Manifest` 子类 + 一个 Loader + 一个 `TrainerBackend`;脊柱全复用。
   - **新模型(同族,如 YOLO)** = 一个 `TrainerBackend` adapter;数据/评估/发布全复用。
   ingest adapter 与 backend 一样经工厂/注册派发,新增不改 caller。

## 理由

- **leverage**:加模型族 = 实现一个 `TrainerBackend` adapter,消融/评估/gate/发布/级联 caller 一行不改。
- **locality**:每族的训练+导出怪癖锁在自己 adapter;级联的取景偏移/门控/回退集中一处。
- **真 seam 非假想**:已有两个 adapter(分类、NanoDet),YOLO 是第三个;`PackagerBackend` 已用同款"按阶段 Protocol"在本仓验证可行。
- **deletion test**:删 `TrainerBackend` → "选哪族怎么训怎么导"的复杂度散回每个 caller;删 `cascade/` → 产品核心逻辑无家(现已散在 results/scripts/)。均"复杂度重现"= 该立。

## 影响 / 后果

- 现有分类/检测代码**包成 adapter,行为不变**(非重写);`eval/ablation/runner` 与 `run_envelope` 改调工厂。
- 检测解锁全流程:能进 `EvalReport`→gate→ModelCard→registry→OTA(成为一等公民)。
- 检测包络分级语义与分类不同(`fp32/int8_sim/field`,无 regional);这是设计点,非缺陷。
- 落地分阶段、非破坏(见实现计划):先立 Protocol + 包 adapter(绿),再泛化 eval/gate,再收 cascade。

## 备选与放弃理由

- **每族全生命周期大对象(形状 A)**:门面顺手,但让"合起来(级联)"无家可归,且 detect 易重新发明 registry/packager;可在形状 B 之上加薄门面,不必作为基座。
- **切到 MMDetection 做"框架式多检测器"**:能统一多检测器,但 MMYOLO=GPL、mmcv 版本耦合,触许可红线(§4)。维持"每检测器独立 repo + subprocess + 统一 ONNX/eval 层"是有意折衷。
- **检测另立一套 DetectionReport 子类**:可行但 publish 链要多态分叉;dict 化 `EvalReport` 更优雅、脊柱零改动。
