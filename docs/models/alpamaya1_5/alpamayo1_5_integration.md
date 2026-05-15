# Alpamayo-1.5 集成说明

本文档描述当前仓库中 `Alpamayo-1.5` 在 `vllm-omni` 内的实际运行方式，重点解释：

- 两阶段拓扑如何组织
- stage 0 的输出如何传到 stage 1
- 最终 `OmniRequestOutput` 里哪些字段来自哪一层
- `stage0_params.n > 1` 时 sample 是如何对齐的

## 1. 总体架构

Alpamayo-1.5 在当前实现里不是单模型直跑，而是一个明确的两阶段 pipeline，入口配置见 [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)：

![vLLM-Omni 两阶段架构图](./arch.png)

1. stage 0 是 `llm` stage
   - `model_arch: Alpamayo1_5Qwen3VLForConditionalGeneration`
   - 负责多模态输入展开、历史轨迹 token 融合、COT 生成、`future_start` 生成

2. stage 1 是 `diffusion` stage
   - `model_arch: Alpamayo1_5TrajectoryPipeline`
   - `engine_input_source: [0]`
   - `final_output_type: trajectory`
   - 消费 stage 0 的文本输出、辅助张量和 KV cache，生成最终轨迹

3. stage 间有两条通道
   - 主数据通道：`custom_process_input_func`
   - KV 通道：`omni_kv_config` connector

## 2. 配置与注册层

当前接入依赖四类注册点：

- stage 配置扩展
  - [vllm_omni/config/stage_config.py](../../../vllm_omni/config/stage_config.py)
  - 支持 `prompt_rewrite_func`、`renderer_rewrite_func`、`request_postprocess_func`、`custom_process_input_func`

- stage 0 模型注册
  - [vllm_omni/model_executor/models/registry.py](../../../vllm_omni/model_executor/models/registry.py)
  - [vllm_omni/model_executor/models/alpamayo1_5/alpamayo1_5_qwen3vl.py](../../../vllm_omni/model_executor/models/alpamayo1_5/alpamayo1_5_qwen3vl.py)

- stage 1 pipeline 注册
  - [vllm_omni/diffusion/registry.py](../../../vllm_omni/diffusion/registry.py)
  - [vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py](../../../vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py)

- HF config 适配
  - [vllm_omni/transformers_utils/configs/alpamayo1_5.py](../../../vllm_omni/transformers_utils/configs/alpamayo1_5.py)

## 3. Stage 0 输入链路

Alpamayo 的 stage 0 输入改写主要集中在 [alpamayo1_5.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py)。

### 3.1 `prompt_rewrite_func`

`rewrite_stage0_prompt_for_vllm_multimodal()` 在 vLLM 做多模态展开前：

- 从 Alpamayo `config.json` 注入 `min_pixels` / `max_pixels`
- 把原始 prompt 文本写入 `additional_information["stage0_prompt_text"]`

### 3.2 `renderer_rewrite_func`

`build_alpamayo_stage0_renderer()` 会提前构造带 Alpamayo 扩展 tokenizer 的 renderer，使 `<|traj_history|>` 等特殊 token 在进入 `InputProcessor` 前就已对齐。

### 3.3 `request_postprocess_func`

`postprocess_stage0_request_for_traj_fusion()` 在 `InputProcessor.process_inputs()` 之后：

- 找到 prompt token ids 中的 `<|traj_history|>` 占位 token
- 用真实的 history trajectory token 替换
- 回写 `request.prompt_token_ids`
- 同步更新 `additional_information["tokenized_data"]`

因此 stage 0 真正送进模型的序列，已经是融合后的最终 token 序列。

## 4. Stage 0 模型与停止语义

[alpamayo1_5_qwen3vl.py](../../../vllm_omni/model_executor/models/alpamayo1_5/alpamayo1_5_qwen3vl.py) 的核心职责有两个：

1. 从 Alpamayo `config.json` 中恢复 stage 0 真实应使用的 Qwen3-VL config
2. 只加载 `vlm.` 前缀权重，避免 stage 1 模块混入 stage 0

停止语义则由 [alpamayo1_5_stop_after_future_start.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5_stop_after_future_start.py) 负责：

1. stage 0 生成 `<|traj_future_start|>`
2. 下一步强制输出 stop token
3. vLLM 在 stop token 处停止

这样做的意义是：

- 生成结果里保留可供 stage 1 rollout 的完整边界
- KV cache 会推进到和原始 HF 路径一致的位置

## 5. Stage 0 输出与 `CompletionOutput` 形成过程

stage 0 最终对外仍然是标准 `RequestOutput`，只是每个 `CompletionOutput` 附带了 Alpamayo 后续要消费的多模态信息。

主要相关文件：

- [vllm_omni/worker/gpu_model_runner.py](../../../vllm_omni/worker/gpu_model_runner.py)
- [vllm_omni/worker/gpu_ar_model_runner.py](../../../vllm_omni/worker/gpu_ar_model_runner.py)
- [vllm_omni/engine/output_processor.py](../../../vllm_omni/engine/output_processor.py)

其中会累计和传递：

- `prompt_mrope_position_delta`
- `attention_mask`
- `initial_noise_x0`
- 其他多模态中间输出
- KV transfer metadata

`OutputProcessor` 会在请求结束时整合这些字段，最终形成带 `CompletionOutput` 列表的 `RequestOutput`。

当 `stage0_params.n = N` 时：

- `RequestOutput.outputs` 中会有 `N` 个 `CompletionOutput`
- 每个 `CompletionOutput` 代表一个 stage 0 sample

这点是后续 stage 1 对齐的起点。

## 6. Stage 0 -> Stage 1 改写

这个阶段的核心函数是 [alpamayo1_5.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py) 里的 `vlm2trajectory()`。

它读取 stage 0 `RequestOutput` 和原始请求 `additional_information`，构造 stage 1 所需上下文。典型字段包括：

- `stage0_prompt_token_ids`
- `stage0_output_token_ids`
- `stage0_sequences`
- `stage0_prompt_length`
- `stage0_output_length`
- `stage0_output_lengths`
- `stage0_num_return_sequences`
- `num_return_sequences`
- `stage0_latent`
- `stage0_latent_shape`
- `initial_noise_x0`
- `stage0_rope_deltas`
- `stage0_attention_mask`
- `stage0_prefill_seq_len`
- `stage0_prefill_seq_lens`
- `stage0_sample_index`

### 6.1 `stage0_sequences`

`stage0_sequences = prompt_token_ids + output_token_ids`

这是 stage 1 恢复 rollout 语境的主依据，必须包含 `future_start`。

### 6.2 `stage0_rope_deltas`

stage 1 实际消费的是 `stage0_rope_deltas`。它来源于 stage 0 输出里的 `prompt_mrope_position_delta`，由 `vlm2trajectory()` 做归一化后传下去。

### 6.3 `initial_noise_x0`

`initial_noise_x0` 无论最初来自原始请求还是中间透传，到了 `vlm2trajectory()` 之后都会明确写入 stage 1 的 `additional_information`。

## 7. 多 sample 扇出与 sample 对齐

这是当前实现里最重要的变化之一。

### 7.1 stage 0 多 sample

当前版本已经完整支持 `stage0_params.n > 1`。实现方式不是让后续流程只看第一个 sample，而是：

- stage 0 先得到多个 `CompletionOutput`
- 引擎侧对这些 sample 做 fanout
- orchestrator 能接住来自 stage 0 的多个 sample
- `vlm2trajectory()` 会为每个 stage 0 sample 各自产生一个 stage 1 prompt

相关实现主要在：

- [vllm_omni/engine/async_omni_engine.py](../../../vllm_omni/engine/async_omni_engine.py)
- [vllm_omni/engine/orchestrator.py](../../../vllm_omni/engine/orchestrator.py)

### 7.2 stage 0 与 stage 1 的一一对应

当前语义是：

- 一个 stage 0 sample
- 对应一个 stage 1 输入 prompt
- 该 prompt 内部可以继续由 stage 1 生成 `M` 个 trajectory sample

如果：

- `stage0_params.n = N`
- `stage1_params.num_outputs_per_prompt = M`

那么最终 shape 是：

- `len(final_output.outputs) == N`
- `pred_xyz.shape == (N, M, T, 3)`
- `pred_rot.shape == (N, M, T, 3, 3)`

其中：

- 第 1 维是 stage 0 sample index
- 第 2 维是 stage 1 sample index

### 7.3 为什么 `cot_text_all` 可能有 `N * M` 条

`cot_token_ids` 的内容来自 stage 0 文本输出，但 stage 1 最终会按 `(stage0_sample, stage1_sample)` 网格组织。

因此：

- 从语义上看，stage 0 每个 sample 本来只有一份 COT
- 但为了和 `pred_xyz/pred_rot` 的 sample 维度对齐，这份 COT 可能在 stage 1 sample 维度上被重复展开

如果调试脚本把它展平，就会看到 `N * M` 条文本，这不表示 sample 错位。

## 8. Stage 1 rollout context

[pipeline_alpamayo1_5.py](../../../vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py) 中，`_prepare_rollout_context()` 是最关键的上下文整理函数。它会从 `additional_information` 和 `sampling_params` 提取：

- `stage0_sequences`
- `past_key_values`
- `stage0_rope_deltas`
- `stage0_attention_mask`
- `initial_noise_x0`
- `stage0_sample_index`

处理过程大致是：

1. 从 `sampling_params.past_key_values` 重建 `DynamicCache`
2. 根据 `stage0_sample_index` 选择本 prompt 对应的 stage 0 上下文切片
3. 校验 sequence 必须包含 `future_start_id`
4. 把 `stage0_rope_deltas` 整理成 rollout 可直接消费的形状
5. 将 `stage0_attention_mask` 与 `initial_noise_x0` 扩展到当前 rollout batch

这一步确保了“第 i 个 stage 0 sample”进入的是“第 i 条 stage 1 rollout 上下文”。

## 9. KV cache 传递

KV cache 不走 `additional_information`，而是走 `sampling_params.past_key_values`。

链路如下：

1. stage 0 在 special token 条件满足时发送 KV
2. diffusion stage 按 `omni_kv_config.need_recv_cache: true` 接收 KV
3. [diffusion_model_runner.py](../../../vllm_omni/diffusion/worker/diffusion_model_runner.py) 在执行前接收 KV
4. `pipeline_alpamayo1_5.py` 把收到的 KV 重建成 `DynamicCache`

因此，stage 1 依赖的是两份输入：

- 文本/张量上下文：来自 `vlm2trajectory()`
- 历史注意力上下文：来自 KV connector

## 10. 最终 `OmniRequestOutput` 的嵌套结构

最终返回对象是 [vllm_omni/outputs.py](../../../vllm_omni/outputs.py) 里的 `OmniRequestOutput`。

当前需要特别注意两层来源：

1. `final_output.custom_output`
   - 主要来自 stage 1
   - 常见字段：
     - `pred_xyz`
     - `pred_rot`
     - `cot_token_ids`
     - `stage0_context_used`

2. `final_output.outputs`、`final_output.prompt_logprobs`、`final_output.prompt_token_ids`
   - 语义上来自 stage 0
   - 通过 `request_output` 或 `_upstream_request_output` 做透传

这意味着：

- 轨迹字段是 stage 1 的
- 文本字段仍然是 stage 0 的

如果你在调试时看到：

- `final_output._upstream_request_output is None`
- 但 `final_output.request_output._upstream_request_output` 非空

这也是正常现象，说明上游 stage 0 的文本输出被保存在内层对象里。

## 11. 当前调试与对照工具

### 11.1 dump 工具

- [request_state_dump.py](../../../vllm_omni/debug/request_state_dump.py)
- [compare_request_state_dump.py](../../../vllm_omni/debug/compare_request_state_dump.py)
- [alpamayo_stage1_rollout_dump.py](../../../vllm_omni/debug/alpamayo_stage1_rollout_dump.py)

适合排查：

- `prompt_token_ids`
- `mrope_position_delta`
- `additional_information`
- stage 1 rollout 输入

### 11.2 推荐脚本

- [alpamoya_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py)
- [alpamoya_2stage_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py)
- [alpamoya_single_batch_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_single_batch_test.py)
- [alpamoya_repro_check.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_repro_check.py)
- [no_rewrite.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/no_rewrite.py)
- [alpamoya_compare_original.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_compare_original.py)
- [alpamoya_fixed_x0_compare.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_fixed_x0_compare.py)

其中 `offline/no_rewrite.py` 现在尤其重要，因为它会同时打印：

- stage 0 文本 sample 数
- `pred_xyz/pred_rot` shape
- `cot_token_ids` 展平后的行为
- request dump / stage1 transition dump 的关键对照项

## 12. 端到端数据流

完整数据流可以按下面顺序理解：

1. 用户传入 prompt
   - `prompt`
   - `multi_modal_data.image`
   - `additional_information`
     - 如 `ego_history_xyz`、`ego_history_rot`、`initial_noise_x0`

2. stage 0 `prompt_rewrite_func`
   - 注入多模态预处理参数

3. stage 0 `renderer_rewrite_func`
   - 构造 Alpamayo tokenizer 对应的 renderer

4. `InputProcessor`
   - 完成多模态展开

5. stage 0 `request_postprocess_func`
   - 把 `<|traj_history|>` 替换成真实 trajectory tokens

6. stage 0 model runner
   - 生成文本 sample
   - 产生 `prompt_mrope_position_delta`、`attention_mask` 等中间信息
   - 按 stop 边界发送 KV

7. `OutputProcessor`
   - 汇总为 stage 0 `RequestOutput`

8. `vlm2trajectory()`
   - 为每个 stage 0 sample 生成一个 stage 1 prompt
   - 写入 `stage0_sample_index`

9. orchestrator
   - 路由 stage 1 prompt
   - 通过 connector 提供 KV

10. stage 1 pipeline
   - 读取 `stage0_sample_index`
   - 选择对应 stage 0 上下文
   - rollout 并输出 `pred_xyz/pred_rot/cot_token_ids`

11. 最终 `OmniRequestOutput`
   - 文本字段保留 stage 0 语义
   - 轨迹字段来自 stage 1 `custom_output`

## 13. 总结

当前 Alpamayo-1.5 接入的核心，不只是“增加一个模型类”，而是完整打通了：

1. stage 0 多模态与 history trajectory 融合
2. stage 0 文本 sample 的标准 `RequestOutput` 化
3. stage 0 到 stage 1 的 sample 级 fanout
4. stage 0 KV cache 到 stage 1 rollout 的传递
5. stage 1 轨迹输出与 stage 0 文本输出在最终 `OmniRequestOutput` 中的统一暴露
