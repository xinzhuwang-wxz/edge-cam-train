"""按族的评估器（[[ADR-0003]] C3）：各产同一个 EvalReport(EnvelopeReport),

经统一的 metrics_from_report → ModelCard → registry 发布链。分类沿用 eval.envelope.build_envelope;
检测见 evaluators.detect.build_detection_report。"""
