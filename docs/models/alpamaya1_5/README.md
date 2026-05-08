# Alpamayo-1.5 快速上手

这份文档面向第一次接手 `Alpamayo-1.5` 的同学，目标是让你快速完成三件事：

1. 确认运行依赖是否齐全
2. 跑通 stage 0-only 和完整两阶段链路
3. 理解当前输出结构与多 sample 语义

这里描述的是当前仓库中的实际实现，不再按某个历史 commit 范围来说明。

## 1. 目录导航

- 总体设计说明：[alpamayo1_5_integration.md](./alpamayo1_5_integration.md)
- 当前关键改动面：[alpamayo1_5_change_inventory.md](./alpamayo1_5_change_inventory.md)
- 两阶段配置文件：[alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)
- 测试目录：[custom_test](../../../tests/diffusion/models/alpamoya/custom_test)
- 离线对照目录：[custom_test/offline](../../../tests/diffusion/models/alpamoya/custom_test/offline)

## 2. 你需要先准备什么

当前文档里提到的 `custom_test` 脚本，大多默认依赖下面几项：

- 模型目录：`/share/models/Alpamayo-1.5-10B`
- 数据集目录：`/share/datasets/ncore_10clips`
- 原始 Alpamayo 源码目录：`../alpamayo1.5/src`

共享路径定义可参考 [common.py](../../../tests/diffusion/models/alpamoya/custom_test/common.py)。

这里需要区分两类依赖：

- `MODEL_PATH` 和 `DATASET_PATH`
  - 对当前大多数 `custom_test` 脚本都是必需的

- `ALPAMAYO_SRC`
  - 主要用于“和原始实现做对照”的脚本
  - 例如 [alpamoya_compare_original.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_compare_original.py)
  - 如果只是验证 `vllm-omni` 自身两阶段链路，不一定每次都需要

如果你的机器路径不同，优先修改：

- `MODEL_PATH`
- `DATASET_PATH`
- `ALPAMAYO_SRC`
- 必要时修改 `CLIP_ID` 和 `T0_US`

## 3. 先理解这套链路在跑什么

当前接入不是单模型直跑，而是两阶段：

1. stage 0
   - 模型类：`Alpamayo1_5Qwen3VLForConditionalGeneration`
   - 职责：处理图像、多模态 prompt、history trajectory token 融合，并生成 COT 与 `future_start`

2. stage 1
   - 模型类：`Alpamayo1_5TrajectoryPipeline`
   - 职责：消费 stage 0 的 token、KV cache 和辅助信息，输出轨迹

3. 两阶段之间
   - 主输入通过 `custom_process_input_func` 转换
   - KV cache 通过 connector 传递

配置入口在 [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)。

### 3.1 最终返回结构

当前两阶段最终返回的是 `OmniRequestOutput`，但里面混合了两类来源不同的数据：

- 文本侧字段
  - 来自 stage 0 的 `RequestOutput`
  - 常见字段：
    - `outputs`
    - `prompt_logprobs`
    - `prompt_token_ids`
    - `finish_reason`
    - `routed_experts`

- 轨迹侧字段
  - 来自 stage 1 diffusion pipeline 的 `custom_output`
  - 常见字段：
    - `pred_xyz`
    - `pred_rot`
    - `cot_token_ids`
    - `stage0_context_used`

实现上，stage 1 最终输出会保留 stage 0 的上游输出引用，因此最终对象里的文本字段仍然代表 stage 0 的结果，而不是 stage 1 重新生成的文本。

### 3.2 当前多 sample 语义

当前版本已经支持 stage 0 多 sample，并且 stage 0 与 stage 1 的 sample 是按顺序一一对应的。

如果：

- `stage0_params.n = N`
- `stage1_params.num_outputs_per_prompt = M`

那么当前输出语义是：

- stage 0 文本 sample 数：`N`
- stage 1 轨迹输出 shape：
  - `pred_xyz.shape == (N, M, T, 3)`
  - `pred_rot.shape == (N, M, T, 3, 3)`

含义是：

- 第 1 维：stage 0 sample index
- 第 2 维：对应这个 stage 0 sample 下，stage 1 生成的 trajectory sample index

例如 `N=2, M=2` 时：

- `final_output.outputs[0]` 对应 `pred_xyz[0, ...]`
- `final_output.outputs[1]` 对应 `pred_xyz[1, ...]`

这表示“每个 stage 0 sample 各自继续生成自己的 stage 1 多轨迹 sample”，而不是把多个 stage 0 sample 混在一起统一采样。

### 3.3 `cot_token_ids` 为什么可能比 stage 0 sample 多

`cot_token_ids` 的内容来自 stage 0 文本输出，但在 stage 1 最终返回时会按 stage 1 sample 网格重新组织。

因此：

- 当 `N=2, M=1` 时，`cot_token_ids` 可理解为 `(2, L)` 或 `(2, 1, L)`
- 当 `N=2, M=2` 时，`cot_token_ids` 可理解为 `(2, 2, L)`

如果调试脚本里把它 `view(-1, L)` 展平，就会看到 `N * M` 条 COT 文本。这是当前表示方式的结果，不代表 sample 对齐错误。

## 4. 最快验证路径

在仓库根目录执行：

```bash
python tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py
```

这一步用于确认 stage 0-only 路径没问题。建议重点关注：

- `stage0_final_output_type`
- `stage0_prompt_contains_fused_history: True`
- 能看到正常的 stage 0 文本输出

接着跑完整两阶段：

```bash
python tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py
```

这一步用于确认完整链路打通。建议重点关注：

- `stage0_context_used`
- `pred_xyz.shape`
- `pred_rot.shape`
- `cot_token_ids.shape`
- `minADE`

如果显式把多 sample 打开，例如：

- `stage0_params.n = 2`
- `stage1_params.num_outputs_per_prompt = 2`

那么更合理的预期是：

- `len(final_output.outputs) == 2`
- `pred_xyz.shape == (2, 2, 64, 3)`
- `pred_rot.shape == (2, 2, 64, 3, 3)`

## 5. 推荐验证顺序

建议按下面顺序验证，定位问题会更快：

1. `tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py`
   - 验证 stage 0-only 是否工作

2. `tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py`
   - 验证两阶段端到端是否工作

3. `tests/diffusion/models/alpamoya/custom_test/alpamoya_single_batch_test.py`
   - 验证 batch 场景是否工作

4. `tests/diffusion/models/alpamoya/custom_test/alpamoya_repro_check.py`
   - 验证重复运行是否稳定

5. `tests/diffusion/models/alpamoya/custom_test/offline/no_rewrite.py`
   - 验证“预编码 prompt + 关闭 stage0 rewrite hook”路径
   - 适合检查 stage 0 输出、stage 1 过渡 payload、最终多 sample 组织是否一致

6. `tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_compare_original.py`
   - 与原始 Alpamayo / HF 路径做中间态对照

7. `tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_fixed_x0_compare.py`
   - 固定 `initial_noise_x0` 后做更严格对比

## 6. 出问题先看什么

常见问题可以先按这张表排：

- stage 0 跑不通
  - 先看模型路径、数据路径、原始 Alpamayo 源码路径是否正确
  - 再跑 `alpamoya_test.py`

- stage 0 能跑，stage 1 没输出
  - 先看 [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml) 里的 `engine_input_source`、`custom_process_input_func`、`omni_kv_config`
  - 再检查 `stage0_context_used`

- stage 1 输出 shape 不对或 KV 上下文异常
  - 先看 `vlm2trajectory()` 产出的 `stage0_sequences`、`stage0_rope_deltas`、`stage0_attention_mask`
  - 再看 `pipeline_alpamayo1_5.py` 的 `_prepare_rollout_context()`

- 多 sample 数量或 sample 对齐异常
  - 先确认 stage 0 文本侧 sample 数是否正确：
    - `len(final_output.outputs)`
  - 再确认 stage 1 prompt 扇出数是否正确：
    - `trajectory_inputs_len`
    - `trajectory_input_sample_indices`
  - 最后确认 trajectory 第一维是否等于 stage 0 sample 数：
    - `pred_xyz.shape[0]`
    - `pred_rot.shape[0]`

- 和原始实现对不齐
  - 跑 `offline/alpamoya_compare_original.py`
  - 必要时启用 request dump / rollout dump

## 7. 调试开关

当前仓库内置两类 dump 工具：

- request state dump
  - 代码：[request_state_dump.py](../../../vllm_omni/debug/request_state_dump.py)

- stage 1 rollout dump
  - 代码：[alpamayo_stage1_rollout_dump.py](../../../vllm_omni/debug/alpamayo_stage1_rollout_dump.py)

如果你要对齐原始实现，推荐直接配合：

- [no_rewrite.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/no_rewrite.py)
- [alpamoya_compare_original.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_compare_original.py)
- [alpamoya_fixed_x0_compare.py](../../../tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_fixed_x0_compare.py)

当前针对多 sample 调试，尤其建议关注这些 dump 字段：

- `stage0_num_return_sequences`
- `stage0_request_outputs_len`
- `stage0_completion_counts`
- `trajectory_inputs_len`
- `trajectory_input_sample_indices`

它们可以帮助判断问题是在：

- stage 0 没生成足够的文本 sample
- `vlm2trajectory()` 没正确扇出到 stage 1
- 还是 stage 1 最终 shape 没按 `(stage0_sample, stage1_sample, ...)` 组织

## 8. 一句话建议

第一次上手时，不要先跑最重的对照脚本。先用 `alpamoya_test.py` 和 `alpamoya_2stage_test.py` 确认链路通，再用 `offline/no_rewrite.py` 和 `offline/alpamoya_compare_original.py` 去对齐细节，效率会高很多。

## 9. 后续关注点

- 继续完善 batch 请求下的统一调试体验
- 减少 stage 0 到 stage 1 之间的冗余字段传输
- 继续补充在线服务路径的验证说明
