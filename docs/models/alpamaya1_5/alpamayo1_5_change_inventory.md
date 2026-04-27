# Alpamayo-1.5 改动文件清单

本文档按 `git diff --name-status f55ea28..28a4791066db609068fbbb8f9f69dd7cd803a928` 整理 `Alpamayo-1.5` 接入 `vllm-omni` 时涉及的文件，并将其划分为：

- 修改文件（`M`）：在现有框架上补齐接入点、注册点和运行时能力
- 新增文件（`A`）：为 Alpamayo 两阶段链路新增模型、配置、调试和测试实现

统计结果如下：

- 修改文件：14 个
- 新增文件：22 个
- 总计：36 个

说明：本清单描述的是上述提交范围内的历史改动目的，不包含当前工作区里对文档本身的后续编辑。

## 1. 修改文件

### 1.1 配置与注册

- `.gitignore`
  - 目的：忽略 `learning/` 目录下的本地实验产物，避免 Alpamayo 定制调试脚本和临时文件污染仓库状态。

- `vllm_omni/config/stage_config.py`
  - 目的：为 stage 配置新增 `prompt_rewrite_func`、`renderer_rewrite_func`、`request_postprocess_func` 三类 hook，并把 Alpamayo 名称映射到对应 stage input processor。
  - 作用：让 YAML 可以声明 stage 0 的 prompt 改写、renderer/tokenizer 替换和 request 后处理逻辑。

- `vllm_omni/engine/arg_utils.py`
  - 目的：注册 `Alpamayo1_5Config`。
  - 作用：让引擎在加载 checkpoint 时能识别 Alpamayo 自定义 HF config，并沿用 Qwen3-VL 的多模态配置。

- `vllm_omni/model_executor/models/registry.py`
  - 目的：注册 `Alpamayo1_5Qwen3VLForConditionalGeneration`。
  - 作用：让 stage 0 可以按 vLLM 模型注册机制加载 Alpamayo 的 VLM backbone 包装类。

- `vllm_omni/diffusion/registry.py`
  - 目的：注册 `Alpamayo1_5TrajectoryPipeline`。
  - 作用：让 diffusion stage 可以按既有 pipeline 查找机制加载 Alpamayo 的轨迹头实现。

- `vllm_omni/entrypoints/async_omni_diffusion.py`
  - 目的：仅在 `model_class_name` 为空时才回退到自动推断的 architecture。
  - 作用：避免入口层把 YAML 中明确指定的 Alpamayo diffusion class 覆盖掉。

### 1.2 引擎初始化与 stage 编排

- `vllm_omni/engine/stage_init_utils.py`
  - 目的：把 stage config 中的 hook 路径解析成可调用函数，并兼容 `engine_input_source` / `input_sources` 的读取。
  - 作用：让 Alpamayo 的自定义输入链路能在 stage 初始化阶段被框架识别。

- `vllm_omni/engine/async_omni_engine.py`
  - 目的：在 stage 0 初始化时支持自定义 renderer；在请求进入引擎时支持 `prompt_rewrite_func` 与 `request_postprocess_func`。
  - 作用：把 Alpamayo 的 tokenizer 替换、图像参数注入、history trajectory token 融合等逻辑接入通用引擎流程。

- `vllm_omni/engine/orchestrator.py`
  - 目的：补充 stage 间原始 prompt 传递点的注释说明。
  - 作用：澄清 `custom_process_input_func` 消费的是 stage 0 输出与原始 prompt 语境，方便维护 `vlm2trajectory()` 这类跨 stage 改写逻辑。

- `vllm_omni/diffusion/diffusion_engine.py`
  - 目的：给 diffusion 请求补上 `need_kv_receive=False` 默认值。
  - 作用：允许不是所有 diffusion 请求都强制接收 KV，为 Alpamayo 之外的路径保留可选行为。

- `vllm_omni/diffusion/worker/diffusion_model_runner.py`
  - 目的：在真正接收 KV cache 之前，先检查 `sampling_params.need_kv_receive`。
  - 作用：让 stage 1 只在需要时接收 stage 0 传来的 KV，避免无条件等待或多余通信。

### 1.3 Runner、输出与调试链路 （具体设备硬件相关）

- `vllm_omni/worker/gpu_model_runner.py`
  - 目的：为 request state 增加可选 dump；兼容不同模型 `get_mrope_input_positions()` 的参数签名；把 `prompt_mrope_position_delta` 写入中间缓冲区。
  - 作用：既保证 Alpamayo 的 mRoPE 位置偏移能流向后续输出，又给对齐原始实现时提供 request-state 级别的调试抓手。

- `vllm_omni/worker/gpu_ar_model_runner.py`
  - 目的：把 `prompt_mrope_position_delta` 从中间缓冲区带到 stage 0 输出 payload。
  - 作用：让 `vlm2trajectory()` 能从 stage 0 输出恢复 `stage0_rope_deltas`，供 stage 1 rollout 使用。

- `vllm_omni/engine/output_processor.py`
  - 目的：为多模态输出累计逻辑增加一组“不应直接 concat”的 key，并统一对外返回 `RequestOutput`。
  - 作用：避免 `attention_mask`、`initial_noise_x0`、`prompt_mrope_position_delta` 等字段在逐步 decode 时被错误拼接，同时保持对 orchestrator 的标准输出形态。

注：如果运行平台切换为 NPU，文件对应关系通常应看 [npu_model_runner.py](../../../vllm_omni/platforms/npu/worker/npu_model_runner.py) 和 [npu_ar_model_runner.py](../../../vllm_omni/platforms/npu/worker/npu_ar_model_runner.py)。不过需要区分“对应文件”和“实际改动落点”：`npu_model_runner.py` 中的 `OmniNPUModelRunner` 继承自 [gpu_model_runner.py](../../../vllm_omni/worker/gpu_model_runner.py) 里的 `OmniGPUModelRunner`，因此像 `request state dump`、`model_intermediate_buffer`、`prompt_mrope_position_delta` 缓冲这类通用逻辑，很多情况下会直接继承生效，不一定需要在 NPU 文件里再改一份；而 `npu_ar_model_runner.py` 拥有独立的 stage 0 输出拼装路径，所以凡是涉及 `payload` 字段补充或 `prompt_mrope_position_delta` 随输出下发的改动，通常都应优先检查这一侧是否需要同步修改。

## 2. 新增文件

### 2.1 文档与部署配置

- [alpamayo1_5_integration.md](./alpamayo1_5_integration.md)
  - 目的：记录 Alpamayo-1.5 接入的整体设计、数据流和关键实现点。

- [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)
  - 目的：定义 Alpamayo 的两阶段部署拓扑。
  - 作用：声明 stage 0 / stage 1 的模型类、hook、KV connector、停止语义和默认 sampling 参数。

### 2.2 Stage 0：VLM backbone 接入

- [vllm_omni/model_executor/models/alpamayo1_5/__init__.py](../../../vllm_omni/model_executor/models/alpamayo1_5/__init__.py)
  - 目的：暴露 stage 0 模型包，供 registry 动态导入。

- [vllm_omni/model_executor/models/alpamayo1_5/alpamayo1_5_qwen3vl.py](../../../vllm_omni/model_executor/models/alpamayo1_5/alpamayo1_5_qwen3vl.py)
  - 目的：实现 Alpamayo stage 0 的 Qwen3-VL 包装类。
  - 作用：从 `vlm_name_or_path` 恢复基础 Qwen3-VL config，保留 Alpamayo 扩展词表，并只加载 `vlm.` 前缀权重。

- [vllm_omni/model_executor/models/alpamayo1_5/runtime.py](../../../vllm_omni/model_executor/models/alpamayo1_5/runtime.py)
  - 目的：沉淀 stage 0 的本地 runtime 基础组件。
  - 作用：集中定义轨迹 special tokens、本地 trajectory tokenizer，以及 history trajectory 的 token 化逻辑。

- [vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py)
  - 目的：实现 Alpamayo stage 0 的输入改写和 stage 0 -> stage 1 的数据转换。
  - 作用：负责 prompt 改写、custom renderer 构造、history trajectory token 融合，以及 `vlm2trajectory()` 输入重写。

- [vllm_omni/model_executor/stage_input_processors/alpamayo1_5_stop_after_future_start.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5_stop_after_future_start.py)
  - 目的：实现 “生成 `<|traj_future_start|>` 后再强制多走一步停止” 的 logits processor。
  - 作用：复现原始 Alpamayo 的 stop 语义，并让发送给 stage 1 的 KV cache 与 HF 路径对齐。

- [vllm_omni/transformers_utils/configs/alpamayo1_5.py](../../../vllm_omni/transformers_utils/configs/alpamayo1_5.py)
  - 目的：定义 `Alpamayo1_5Config`。
  - 作用：让 Alpamayo checkpoint 在 HF/vLLM 看起来像 Qwen3-VL 配置，同时保留自身附加字段。

### 2.3 Stage 1：Trajectory pipeline 接入

- [vllm_omni/diffusion/models/alpamayo1_5/__init__.py](../../../vllm_omni/diffusion/models/alpamayo1_5/__init__.py)
  - 目的：暴露 stage 1 diffusion 模型包，供 registry 动态导入。

- [vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py](../../../vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py)
  - 目的：实现 Alpamayo stage 1 轨迹头的主 pipeline。
  - 作用：接收 stage 0 的 token、KV cache 和辅助张量，构建 rollout context，并输出 `pred_xyz`、`pred_rot`、`cot_token_ids`。

- [vllm_omni/diffusion/models/alpamayo1_5/runtime.py](../../../vllm_omni/diffusion/models/alpamayo1_5/runtime.py)
  - 目的：沉淀 stage 1 的运行时数学和基础模块。
  - 作用：提供 action space、flow matching、projection 模块，以及轨迹平滑与约束求解工具，降低主 pipeline 文件耦合度。

### 2.4 调试工具

- [vllm_omni/debug/__init__.py](../../../vllm_omni/debug/__init__.py)
  - 目的：声明 `vllm_omni.debug` 包。

- [vllm_omni/debug/request_state_dump.py](../../../vllm_omni/debug/request_state_dump.py)
  - 目的：提供按环境变量启用的 request-state dump 工具。
  - 作用：在 request 初始化和入 batch 等阶段导出 `prompt_token_ids`、`mrope_position_delta`、`mm_features`、`additional_information` 等中间态。

- [vllm_omni/debug/compare_request_state_dump.py](../../../vllm_omni/debug/compare_request_state_dump.py)
  - 目的：比较两份 request-state dump。
  - 作用：用于定位 vLLM-Omni 路径与原始 Alpamayo/HF 路径在中间状态上的差异。

- [vllm_omni/debug/alpamayo_stage1_rollout_dump.py](../../../vllm_omni/debug/alpamayo_stage1_rollout_dump.py)
  - 目的：提供 stage 1 rollout payload 的按需 dump 工具。
  - 作用：把 `initial_noise_x0`、`attention_mask`、`position_ids`、step0 输入等 rollout 关键上下文落盘，便于精确对比。

### 2.5 测试与对照脚本

- [tests/diffusion/models/alpamoya/custom_test/common.py](../../../tests/diffusion/models/alpamoya/custom_test/common.py)
  - 目的：抽取共享的数据加载、prompt 构造、YAML 生成和 sampling params 构造逻辑。

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py)
  - 目的：验证 stage 0-only 路径能独立跑通。
  - 作用：检查 history token 是否已融合、stage 0 是否正常输出 COT 与 stop 结果。

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py)
  - 目的：验证完整两阶段链路能端到端运行。
  - 作用：检查最终 trajectory 输出 shape、COT token 和 minADE 等结果。

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_compare_original.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_compare_original.py)
  - 目的：把 vllm-omni 两阶段路径与原始 Alpamayo/HF 推理逐步对照。
  - 作用：同时比较 request-state dump、stage 1 transition dump 和最终输出张量。

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_fixed_x0_compare.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_fixed_x0_compare.py)
  - 目的：在固定 `initial_noise_x0` 的条件下做更稳定的输出对比。
  - 作用：减少噪声来源，聚焦 stage 1 rollout context 和结果差异。

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_repro_check.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_repro_check.py)
  - 目的：检查同一配置下重复运行的可复现性。
  - 作用：比较两次执行的 `pred_xyz`、`pred_rot` 和 `cot_token_ids` 是否一致。

- [tests/diffusion/models/alpamoya/custom_test/alpamoya_single_batch_test.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_single_batch_test.py)
  - 目的：验证一次 Omni 调用中处理多 prompt / batch 输入的场景。
  - 作用：确认两阶段链路在 batch 模式下的输出形状、上下文传递和 ADE 计算正常。

## 3. 可以怎样配合阅读

如果想从“为什么要改”理解这批文件，建议按下面顺序读：

1. 先看 [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)，理解两阶段拓扑和 hook/connector 配置。
2. 再看 [alpamayo1_5.py](../../../vllm_omni/model_executor/stage_input_processors/alpamayo1_5.py) 与 [pipeline_alpamayo1_5.py](../../../vllm_omni/diffusion/models/alpamayo1_5/pipeline_alpamayo1_5.py)，把 stage 0 与 stage 1 的主数据流串起来。
3. 最后看 [vllm_omni/engine](../../../vllm_omni/engine)、 [vllm_omni/worker](../../../vllm_omni/worker) 和 [tests/diffusion/models/alpamoya/custom_test](../../../tests/diffusion/models/alpamoya/custom_test)，理解这些通用框架改动是如何支撑 Alpamayo 接入与验证的。
