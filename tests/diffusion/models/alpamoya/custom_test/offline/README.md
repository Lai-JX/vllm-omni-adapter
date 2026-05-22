# Alpamayo offline custom tests

这个目录用于离线调试 Alpamayo 1.5 reference/original 实现与 vllm-omni 两阶段实现之间的差异，重点覆盖：

- stage0 / stage1 两阶段联调
- fixed `x0` 复现与对比
- reference KV / omni KV 互换实验
- structured dump 落盘与逐字段比较
- 多样本批量对比
- 基础可复现性与单阶段验证

## 依赖与前提

默认依赖以下本地环境：

- 模型路径：`/share/models/Alpamayo-1.5-10B`
- 数据集路径：`/share/datasets/ncore_10clips`
- `alpamayo1.5` 源码目录已存在于 workspace 同级目录
- 当前 Python 环境能导入：
  - `alpamayo1_5`
  - `vllm`
  - `vllm_omni`
  - `torch`
  - `PIL`

公共配置和数据加载逻辑集中在 [common.py](common.py)。

## 目录内核心脚本

### 1. [alpamoya_compare_original.py](alpamoya_compare_original.py)
主对比 harness，负责封装：

- `run_original(...)`
- `run_omni(...)`
- fixed `x0` 构造
- stage0/stage1 structured dump
- tensor compare
- compare summary 落盘
- reference / omni KV dump 与重放辅助逻辑

大多数专项脚本都基于这个文件里的 helper 组装。

### 2. [alpamoya_fixed_x0_compare.py](alpamoya_fixed_x0_compare.py)
用途：

- 固定同一份 `initial_noise_x0`
- 直接比较完整 `original` vs `omni`
- 输出 `minADE`、`pred_xyz/pred_rot` 对比
- 对比 stage1 transition / rollout context / rollout step / internal dumps

适合用作当前最标准的单样本 reference-vs-omni 离线对比入口。

运行示例：

```bash
python tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_fixed_x0_compare.py
```

### 3. [alpamayo_fixed_kv_x0_compare.py](alpamayo_fixed_kv_x0_compare.py)
用途：

- 先跑 reference/original
- 将 reference 生成的 KV 提供给 omni
- 观察 `omni_with_reference_kv` 与 original 的收敛程度

用于回答：

- 如果 omni 使用 reference 的 KV，后续 stage1 输出是否接近 reference？

运行示例：

```bash
python tests/diffusion/models/alpamoya/custom_test/offline/alpamayo_fixed_kv_x0_compare.py
```

### 4. [alpamoya_fixed_omni_kv_x0_compare.py](alpamoya_fixed_omni_kv_x0_compare.py)
用途：

- 固定 `x0`
- 以 omni 产出的 KV 为基准做对比
- 主要用于观察 omni KV 在后续 stage1 中带来的分叉

适合继续排查：

- divergence 是否跟随 omni KV 传播
- stage1 expert/internal dump 从哪个位置开始分叉

### 5. [alpamoya_multi_sample_compare.py](alpamoya_multi_sample_compare.py)
用途：

- 在多个 clip 上批量比较 `original` vs `omni`
- 支持固定 `x0`
- 支持自定义 diffusion steps
- 输出每个 case 的：
  - `original_minADE`
  - `omni_minADE`
  - `delta`
  - `omni_worse`
  - `cot_equal`
- 最后汇总统计：
  - `omni_worse_count`
  - `omni_better_count`
  - `equal_count`
  - `cot_equal_count`

当前执行方式是：

1. 先跑所有 original/reference
2. 清理显存
3. 再统一启动一次 omni 跑所有 case

这样可以减少 original 与 omni 交替执行导致的 OOM 风险。

运行示例：

```bash
python tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_multi_sample_compare.py --limit 5
```

带 fixed `x0` 和 step 参数：

```bash
python tests/diffusion/models/alpamoya/custom_test/offline/alpamoya_multi_sample_compare.py \
  --limit 5 \
  --fixed-x0 \
  --steps 20
```

### 6. [alpamoya_2stage_test.py](alpamoya_2stage_test.py)
用途：

- 最小化验证 AsyncOmni 两阶段推理是否能跑通
- 打印：
  - `stage0_context_used`
  - `pred_xyz.shape`
  - `pred_rot.shape`
  - `cot_token_ids.shape`
  - `minADE`

适合作为最简单的两阶段烟雾测试入口。

### 7. [alpamoya_test.py](alpamoya_test.py)
用途：

- 只跑 stage0
- 打印 stage0 文本输出和基础元信息

适合快速检查：

- prompt 是否正常
- stage0 token 输出是否正常
- fused history 是否进入 prompt

### 8. [alpamoya_repro_check.py](alpamoya_repro_check.py)
用途：

- 用同一套输入连续跑两次 omni
- 对比 `pred_xyz` / `pred_rot` / `cot_token_ids`

适合验证：

- 当前环境下同配置是否可复现

### 9. [alpamoya_single_batch_test.py](alpamoya_single_batch_test.py)
用途：

- 一次 Omni 调用里塞多个 Alpamayo prompt 做 batch 测试
- 适合检查多请求批处理行为

### 10. 其它脚本

- [next_step_after_special_token.py](next_step_after_special_token.py)
- [no_rewrite.py](no_rewrite.py)

这两个更偏实验性调试脚本，通常在已有明确问题时按需使用。

## 常见产物

多数 compare 脚本会在当前工作目录下生成：

- `alpamayo_compare_artifacts/run_<timestamp>/`

目录内通常包含：

- `stage1_transition`
- `stage1_rollout_context`
- `stage1_rollout_step`
- `action_in_proj_internal`
- `expert_internal`
- compare summary
- reference / omni KV dump

这些产物主要用于进一步回答：

- stage0 token 是否一致
- rollout context 是否一致
- expert 输入输出是否一致
- divergence 第一次出现在什么位置

## 建议使用顺序

如果只是从头定位问题，推荐按这个顺序：

1. [alpamoya_2stage_test.py](alpamoya_2stage_test.py)  
   先确认两阶段流程能正常跑通。
2. [alpamoya_fixed_x0_compare.py](alpamoya_fixed_x0_compare.py)  
   做标准的单样本 original vs omni 对比。
3. [alpamayo_fixed_kv_x0_compare.py](alpamayo_fixed_kv_x0_compare.py)  
   看 omni 使用 reference KV 后是否收敛。
4. [alpamoya_fixed_omni_kv_x0_compare.py](alpamoya_fixed_omni_kv_x0_compare.py)  
   看 divergence 是否跟随 omni KV。
5. [alpamoya_multi_sample_compare.py](alpamoya_multi_sample_compare.py)  
   验证问题是否只出现在个别样本，还是整体分布偏移。

## 备注

- 这个目录主要服务于调试和分析，不是长期产品化接口。
- 许多脚本默认会读写本地 artifact，并依赖当前工作目录。
- 如果你修改了 [alpamoya_compare_original.py](alpamoya_compare_original.py)，通常会影响多个专项脚本，因为它是这里的共享 harness。
