# Alpamayo-1.5 当前改动面清单

这份清单不再按某个固定 commit 范围统计“新增了多少文件、删了多少行”，而是按当前仓库中的实际职责来整理 Alpamayo-1.5 相关改动面，方便排查和继续演进。

## 1. 配置与注册

### 1.1 两阶段配置

- [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)
  - 作用：定义 stage 0 / stage 1 拓扑、hook、KV connector、停止语义和默认 sampling 参数

### 1.2 模型与 pipeline 注册

- [vllm_omni/model_executor/models/registry.py](../../../vllm_omni/model_executor/models/registry.py)
  - 作用：注册 `Alpamayo1_5Qwen3VLForConditionalGeneration`

- [vllm_omni/diffusion/registry.py](../../../vllm_omni/diffusion/registry.py)
  - 作用：注册 `Alpamayo1_5TrajectoryPipeline`

- [vllm_omni/transformers_utils/configs/alpamayo1_5.py](../../../vllm_omni/transformers_utils/configs/alpamayo1_5.py)
  - 作用：让引擎能识别 Alpamayo 的 HF config，并恢复其 Qwen3-VL 基座语义

## 2. Stage 0 相关改动

### 2.1 模型包装与本地 runtime

- [vllm_omni/model_executor/models/alpamayo1_5/alpamayo1_5_qwen3vl.py](../../../vllm_omni/model_executor/models/alpamayo1_5/alpamayo1_5_qwen3vl.py)
  - 作用：恢复 stage 0 使用的 Qwen3-VL config，并过滤只属于 `vlm.` 的权重

- [vllm_omni/model_executor/models/alpamayo1_5/runtime.py](../../../vllm_omni/model_executor/models/alpamayo1_5/runtime.py)
  - 作用：集中沉淀 trajectory special tokens 与 history trajectory tokenization 逻辑

### 2.2 输入改写与停止语义

- [vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py)
  - 作用：负责 `prompt_rewrite_func`、`renderer_rewrite_func`、`request_postprocess_func`，以及 `vlm2trajectory()`

- [vllm_omni/model_executor/stage_input_processors/alpamayo1_5_stop_after_future_start.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5_stop_after_future_start.py)
  - 作用：复现 `<|traj_future_start|>` 后再强制停止的语义，使 KV 边界与原始实现对齐

### 2.3 多 sample 输出起点

- [vllm_omni/engine/output_processor.py](../../../vllm_omni/engine/output_processor.py)
  - 作用：把 stage 0 decode 结果整理成标准 `RequestOutput`，每个 sample 对应一个 `CompletionOutput`

- [vllm_omni/worker/gpu_model_runner.py](../../../vllm_omni/worker/gpu_model_runner.py)
  - 作用：补充 mRoPE、multimodal buffer 和 request dump 所需中间态

- [vllm_omni/worker/gpu_ar_model_runner.py](../../../vllm_omni/worker/gpu_ar_model_runner.py)
  - 作用：把 `prompt_mrope_position_delta` 等关键信息带入 stage 0 输出 payload

## 3. Stage 0 -> Stage 1 过渡改动

### 3.1 fanout 与 sample 对齐

- [vllm_omni/engine/async_omni_engine.py](../../../vllm_omni/engine/async_omni_engine.py)
  - 作用：支持 stage 0 `n > 1` 时的 child request fanout，使多个 stage 0 sample 都能继续流向 stage 1

- [vllm_omni/engine/orchestrator.py](../../../vllm_omni/engine/orchestrator.py)
  - 作用：接住 stage 0 的多 sample 输出，并把 `custom_process_input_func` 的多个结果正确路由到 stage 1

### 3.2 中间字段组织

- [vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py)
  - 作用：在 `vlm2trajectory()` 中生成：
    - `stage0_sequences`
    - `stage0_rope_deltas`
    - `stage0_attention_mask`
    - `initial_noise_x0`
    - `stage0_sample_index`

这部分是“stage 0 sample 是否和 stage 1 sample 一一对应”的关键。

## 4. Stage 1 相关改动

### 4.1 pipeline 与 runtime

- [vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py](../../../vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py)
  - 作用：根据 stage 0 输出重建 rollout context，生成 `pred_xyz`、`pred_rot`、`cot_token_ids`

- [vllm_omni/diffusion/models/alpamayo1_5/runtime.py](../../../vllm_omni/diffusion/models/alpamayo1_5/runtime.py)
  - 作用：沉淀 trajectory diffusion 所需的 action space、flow matching 和辅助数学逻辑

### 4.2 diffusion 请求聚合与 KV

- [vllm_omni/diffusion/diffusion_engine.py](../../../vllm_omni/diffusion/diffusion_engine.py)
  - 作用：负责 batch diffusion 结果聚合，并保留 custom output / latents

- [vllm_omni/entrypoints/async_omni_diffusion.py](../../../vllm_omni/entrypoints/async_omni_diffusion.py)
  - 作用：负责 batch 入口返回的一致性，并避免显式 class name 被自动 architecture 推断覆盖

- [vllm_omni/diffusion/stage_diffusion_client.py](../../../vllm_omni/diffusion/stage_diffusion_client.py)
  - 作用：把 stage 1 diffusion 输出包装成可并回总链路的结果

- [vllm_omni/diffusion/worker/diffusion_model_runner.py](../../../vllm_omni/diffusion/worker/diffusion_model_runner.py)
  - 作用：按需接收 stage 0 传来的 KV cache

## 5. 最终输出包装

- [vllm_omni/outputs.py](../../../vllm_omni/outputs.py)
  - 作用：定义 `OmniRequestOutput`
  - 当前关键点：
    - 文本字段透传 stage 0 `RequestOutput`
    - 轨迹字段暴露 stage 1 `custom_output`
    - 通过 `request_output` / `_upstream_request_output` 支持嵌套透传

这部分直接决定了为什么最终对象里会同时看到：

- `outputs`, `prompt_logprobs`, `finish_reason`
- `pred_xyz`, `pred_rot`, `cot_token_ids`

## 6. 调试与对照工具

### 6.1 dump 工具

- [vllm_omni/debug/request_state_dump.py](../../../vllm_omni/debug/request_state_dump.py)
  - 作用：导出 request 初始化、入 batch 等阶段的中间态

- [vllm_omni/debug/compare_request_state_dump.py](../../../vllm_omni/debug/compare_request_state_dump.py)
  - 作用：比较两份 request dump

- [vllm_omni/debug/alpamayo_stage1_rollout_dump.py](../../../vllm_omni/debug/alpamayo_stage1_rollout_dump.py)
  - 作用：导出 stage 1 rollout payload

### 6.2 测试与对照脚本

- [tests/diffusion/models/alpamoya/custom_test/common.py](../../../tests/diffusion/models/alpamoya/custom_test/common.py)
  - 作用：共享路径、数据加载和 sampling params 构造逻辑

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py)
  - 作用：验证 stage 0-only 路径

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py)
  - 作用：验证两阶段端到端链路

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_single_batch_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_single_batch_test.py)
  - 作用：验证 batch 输入

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_repro_check.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_repro_check.py)
  - 作用：验证复现性

- [tests/diffusion/models/alpamoya/custom_test/offline/no_rewrite.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/no_rewrite.py)
  - 作用：验证“预编码 prompt + 关闭 rewrite hook”路径，并打印 stage 0 / stage 1 多 sample 对齐信息

- [tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_compare_original.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_compare_original.py)
  - 作用：把当前 `vllm-omni` 路径与原始 Alpamayo / HF 路径逐步对照

- [tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_fixed_x0_compare.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_fixed_x0_compare.py)
  - 作用：固定 `initial_noise_x0` 后做更稳定的对比

## 7. 建议阅读顺序

如果想快速理解当前代码面，建议按下面顺序：

1. 先看 [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)
   - 理解两阶段拓扑和 hook / connector 配置

2. 再看 [alpamayo1_5.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py)
   - 理解 stage 0 输入改写和 `vlm2trajectory()` 过渡逻辑

3. 再看 [pipeline_alpamayo1_5.py](../../../vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py)
   - 理解 stage 1 如何消费 stage 0 上下文

4. 最后看 [outputs.py](../../../vllm_omni/outputs.py)、[orchestrator.py](../../../vllm_omni/engine/orchestrator.py)、[no_rewrite.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/no_rewrite.py)
   - 理解最终输出包装、多 sample 对齐和调试方式
