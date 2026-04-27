# Alpamayo-1.5 快速上手
> 当前版本基于 `f55ea28(v0.18.0)` 修改而来。  
> 基础镜像：vllm/vllm-omni:v0.18.0

这份文档面向第一次接手 `Alpamayo-1.5` 的同学，目标是让你快速完成三件事：

1. 确认运行依赖是否齐全
2. 跑通 stage 0-only 和完整两阶段链路
3. 知道出问题时应该先看哪里

如果你只想最快验证接入是否正常，直接看下面的“5 分钟跑通”即可。

## 1. 目录导航

- 总体设计说明：[alpamayo1_5_integration.md](./alpamayo1_5_integration.md)
- 改动文件清单：[alpamayo1_5_change_inventory.md](./alpamayo1_5_change_inventory.md)
- 两阶段配置文件：[alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml)
- 测试目录：[custom_test](../../../tests/diffusion/models/alpamoya/custom_test)

## 2. 你需要先准备什么

当前文档里提到的 `custom_test` 脚本，大多默认依赖下面几项：

- 模型目录：`/share/models/Alpamayo-1.5-10B`
- 数据集目录：`/share/datasets/ncore_10clips`
- 原始 Alpamayo 源码目录：`../alpamayo1.5/src`

对应代码位置见 [common.py](../../../tests/diffusion/models/alpamoya/custom_test/common.py#L22)。

这里需要特别区分两类依赖：

- `MODEL_PATH` 和 `DATASET_PATH`
  - 对当前 `custom_test` 里的运行脚本基本都是必需的

- `ALPAMAYO_SRC`
  - 不是所有测试脚本、也不是所有部署场景的硬性前提
  - 只有当脚本直接或间接依赖原始 `alpamayo1_5` Python 包时才需要
  - 当前最典型的依赖入口是 [common.py](../../../tests/diffusion/models/alpamoya/custom_test/common.py) 和 [alpamoya_compare_original.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_compare_original.py)

换句话说，`ALPAMAYO_SRC` 更像是“测试辅助 / 原始实现对照”依赖，而不是这套两阶段接入逻辑本身的统一前置条件。

如果你的机器路径不同，优先修改这里：

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

配置入口在 [alpamayo1_5.yaml](../../../vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml#L1)。

## 4. 5 分钟跑通

在仓库根目录执行：

```bash
python tests/diffusion/models/alpamoya/custom_test/alpamoya_test.py
```

这一步用于确认 stage 0-only 路径没问题。你应该重点关注：

- `stage0_final_output_type`
- `stage0_prompt_contains_fused_history: True`
- 能看到正常的 stage 0 文本输出

接着跑完整两阶段：

```bash
python tests/diffusion/models/alpamoya/custom_test/alpamoya_2stage_test.py
```

这一步用于确认完整链路打通。你应该重点关注：

- `stage0_context_used`
- `pred_xyz.shape`
- `pred_rot.shape`
- `cot_token_ids.shape`
- `minADE`

如果这两步都正常，说明最核心的部署链路已经跑起来了。

## 5. 推荐验证顺序

建议不要一上来就跑最重的对照脚本，按这个顺序会更省时间：

1. `alpamoya_test.py`
   - 验证 stage 0-only 是否工作

2. `alpamoya_2stage_test.py`
   - 验证两阶段端到端是否工作

3. `alpamoya_single_batch_test.py`
   - 验证 batch 场景是否工作

4. `alpamoya_repro_check.py`
   - 验证重复运行是否稳定

5. `alpamoya_compare_original.py`
   - 与原始 Alpamayo / HF 路径做中间态对照

6. `alpamoya_fixed_x0_compare.py`
   - 固定 `initial_noise_x0` 后做更严格对比

这些脚本都在 [tests/diffusion/models/alpamoya/custom_test](../../../tests/diffusion/models/alpamoya/custom_test)。

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

- 和原始实现对不齐
  - 跑 `alpamoya_compare_original.py`
  - 必要时启用 request dump / rollout dump

## 7. 调试开关

当前仓库已经内置两类 dump 工具：

- request state dump
  - 代码：[request_state_dump.py](../../../vllm_omni/debug/request_state_dump.py)

- stage 1 rollout dump
  - 代码：[alpamayo_stage1_rollout_dump.py](../../../vllm_omni/debug/alpamayo_stage1_rollout_dump.py)

如果你要对齐原始实现，推荐直接配合：

- [alpamoya_compare_original.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_compare_original.py)
- [alpamoya_fixed_x0_compare.py](../../../tests/diffusion/models/alpamoya/custom_test/alpamoya_fixed_x0_compare.py)

## 8. 一句话建议

第一次上手时，不要先读完整改动清单，也不要一开始就跑最重的 compare 脚本。先用 `alpamoya_test.py` 和 `alpamoya_2stage_test.py` 确认链路通，再按问题去读 [alpamayo1_5_integration.md](./alpamayo1_5_integration.md) 和 [alpamayo1_5_change_inventory.md](./alpamayo1_5_change_inventory.md)，效率会高很多。
