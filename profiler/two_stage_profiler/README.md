# two_stage_profiler

用于测量两阶段链路（stage0 LLM -> KV transfer -> stage1 diffusion）在不同 `sample_count` / token 规模下的时延，并生成静态图与时间线可视化。

## 目录内容

- [sweep_two_stage_profile_vs_tokens.py](sweep_two_stage_profile_vs_tokens.py)
  - 主采集脚本。
  - 复用 tokenized 输入构造逻辑，按 `sample_count` 逐组拼接样本，运行完整 two-stage pipeline，输出 JSON / CSV / service log / torch trace。
- [plot_two_stage_profile_vs_tokens.py](plot_two_stage_profile_vs_tokens.py)
  - 从 metrics JSON 生成 5 张静态指标图。
- [render_two_stage_timestamp_timeline.py](render_two_stage_timestamp_timeline.py)
  - 从 metrics JSON 生成交互式 HTML 时间线，用于查看各阶段时间戳和 phase gap。
- [log/](log/)
  - 每次 sweep 的输出目录，按 `two_stage_profile_vs_tokens_<timestamp>` 组织。

## 采集脚本说明

主脚本是 [sweep_two_stage_profile_vs_tokens.py](sweep_two_stage_profile_vs_tokens.py)。

它做的事情大致是：

1. 从 `profiler/batch_sweep_cont_inproc_tokenized_warmup.py` 复用样本构造逻辑。
2. 读取 `profiler/alpamayo1_5_gpu0.yaml`，生成运行时 two-stage 配置。
3. 为不同 `sample_count` 构造 merged request。
4. 可选执行一次 warmup，且 warmup 样本不会复用于正式测量。
5. 运行完整 stage0 -> stage1 流程。
6. 从最终输出和 service log 中回填：
   - stage0 profile
   - KV send / receive 时间
   - stage submit 时间
   - diffusion 时间
7. 输出结构化结果到 `metrics/`。

### 常用环境变量

脚本主要通过环境变量控制：

- `BS`
  - profiler 中记录的 batch size 标签，默认 `1`
- `CLIP_START`
  - 从候选 clip 列表的哪个位置开始取样，默认 `0`
- `CLIP_CHUNK`
  - 只使用某个 chunk 的 clip，默认 `3116`
- *`REPEAT` ⭐
  - 每个 `sample_count` 重复次数，默认 `1`
- `SKIP_WARMUP`
  - 是否跳过 warmup，默认 `0`
- `ENABLE_PREFIX_CACHING`
  - 是否开启 stage0 prefix caching，默认 `0`
- `SAMPLE_COUNT_LIST`⭐
  - 要测试的 sample_count 列表，默认 `1,2,4,8,16,32`

### 运行示例

```bash
python /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/sweep_two_stage_profile_vs_tokens.py
```

指定参数示例：

```bash
BS=1 \
REPEAT=3 \
SAMPLE_COUNT_LIST=1,2,4,8 \
CLIP_START=0 \
CLIP_CHUNK=3116 \
ENABLE_PREFIX_CACHING=0 \
python /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/sweep_two_stage_profile_vs_tokens.py
```

## 输出目录结构

每次运行会生成：

```text
log/
  two_stage_profile_vs_tokens_<timestamp>/
    alpamayo1_5_gpu0_two_stage_profile_vs_tokens.yaml
    metrics/
      two_stage_profile_vs_tokens_metrics.json
      two_stage_profile_vs_tokens_metrics.csv
      two_stage_profile_vs_tokens_engine_metrics_bs_<BS>.csv
    svc_logs/
      two_stage_profile_vs_tokens_service.log
    torch_traces/
      ...
```

说明：

- `metrics.json`
  - 最完整的结构化结果，后续画图和时间线渲染都基于它。
- `metrics.csv`
  - 便于快速筛查、导表分析。
- `engine_metrics_bs_<BS>.csv`
  - 引擎侧原始统计。
- `service.log`
  - 用于回填 stage submit / KV / diffusion 时间戳。
- `torch_traces/`
  - torch profiler trace 输出。

## 关键指标与时间戳

`sweep_two_stage_profile_vs_tokens.py` 会输出如下几类核心字段。

### 请求级基础字段

- `request_id`
- `sample_count`
- `clip_ids`
- `repeat_index`
- `total_prompt_tokens`
- `total_postprocess_prompt_tokens`
- `mean_prompt_tokens_per_sample`
- `batch_total_scheduled_tokens`
- `output_tokens`

### 端到端与分段时延

- `lat`
  - 整个请求 wall-clock latency
- `stage0_llm_ms`
  - stage0 总生成时间
- `embed_multimodal_ms`
  - stage0 embed 时间
- `forward_ms`
  - stage0 forward 时间
- `subsequent_decode_ms`
  - `forward_end -> kv_s0_start_time` 的 wall-clock 间隔
- `kv_transfer_ms`
  - 当前等于 `kv_tran_total`
- `kv_tran_total`
  - `(kv_s1_end_time - kv_s0_start_time) * 1000`
- `s1_diffusion_ms`
  - stage1 diffusion 总时间
- `diffusion_ms`
  - 当前与 `s1_diffusion_ms` 对齐

### KV 相关字段

- `kv_s0_start_time`
- `kv_s0_end_time`
- `kv_s0_extract_ms`
- `kv_s0_transfer_only_ms`
- `kv_s0_extract_plus_transfer_ms`
- `kv_s1_receive_start_time`
- `kv_s1_end_time`
- `kv_s1_receive_ms`
- `kv_s1_tran_ms`
- `kv_s1_prep_ms`

### stage submit / diffusion 时间戳

- `stage0_submit_start`
- `stage0_submit_end`
- `stage1_submit_start`
- `stage1_submit_end`
- `stage1_diffusion_start_time`
- `stage1_diffusion_end_time`

### stage0 profile 时间戳

- `embed_start`
- `embed_end`
- `forward_start`
- `forward_end`
- `start`
- `now`

## 静态画图脚本

使用 [plot_two_stage_profile_vs_tokens.py](plot_two_stage_profile_vs_tokens.py) 从 metrics JSON 生成汇总图。

### 用法

```bash
python /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/plot_two_stage_profile_vs_tokens.py \
  /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/log/<run>/metrics/two_stage_profile_vs_tokens_metrics.json
```

也可以指定输出路径和标题：

```bash
python /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/plot_two_stage_profile_vs_tokens.py \
  /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/log/<run>/metrics/two_stage_profile_vs_tokens_metrics.json \
  --output /tmp/two_stage_profile.png \
  --title "Two-stage profile"
```

### 当前画出的 5 个指标

1. `image_count` vs `embed_multimodal_ms`
2. `batch_total_scheduled_tokens` vs `forward_ms`
3. `batch_total_scheduled_tokens` vs `subsequent_decode_ms / (output_tokens / sample_count)`
4. `batch_total_scheduled_tokens` vs `kv_transfer_ms`
5. `batch_total_scheduled_tokens` vs `s1_diffusion_ms`

说明：

- 第 3 张图按单个 decode 归一化，便于比较不同 `sample_count` 下的 subsequent decode 开销。
- 图中会叠加 raw points 与 `mean ± std`。

## 时间线 HTML 渲染脚本

使用 [render_two_stage_timestamp_timeline.py](render_two_stage_timestamp_timeline.py) 从 metrics JSON 生成交互式 HTML。

### 用法

```bash
python /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/render_two_stage_timestamp_timeline.py \
  /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/log/<run>/metrics/two_stage_profile_vs_tokens_metrics.json
```

默认输出到：

```text
<json>.timeline.html
```

也可以指定输出文件：

```bash
python /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/render_two_stage_timestamp_timeline.py \
  /workspace/project/RL-learning/vllm-omni/profiler/two_stage_profiler/log/<run>/metrics/two_stage_profile_vs_tokens_metrics.json \
  --output /tmp/two_stage_timeline.html
```

### 当前时间线 phase

- `stage0_submit`
- `embed`
- `forward`
- `subsequent_decode`
- `kv_send`
- `stage1_submit`
- `stage1_queue`
- `kv_recv`
- `stage1_gap`
- `diffusion`

其中：

- `stage1_queue = stage1_submit_start -> kv_s1_receive_start_time`
- `stage1_gap = kv_s1_end_time -> stage1_diffusion_start_time`

这两段通常最适合用来定位“KV 已经发完 / 收完，但 stage1 还没真正开始算”的等待时间。

## 常见分析口径

### 1. subsequent decode

当前定义为：

```text
forward_end -> kv_s0_start_time
```

即 stage0 forward 结束后，到 stage0 开始做 KV extract / send 之间的 wall-clock 间隔。

### 2. kv_transfer_ms

当前定义为：

```text
(kv_s1_end_time - kv_s0_start_time) * 1000
```

这是端到端 KV span，不只是 send/recv 算子本身，因此会包含中间调度/排队时间。

### 3. 为什么 `output_tokens / sample_count` 更接近单样本 decode 次数

`output_tokens` 统计的是展平后的总 token 数；在 merged request 下它通常等于所有 sample 的总和，所以做单 decode 对比时常用：

```text
output_tokens / sample_count
```

## 注意事项

- warmup 样本不会复用于正式测量。
- `sample_count` 越大，`total_postprocess_prompt_tokens` 也越大，可能超过 `max_model_len` 被跳过。
- 时间线 HTML 依赖 metrics JSON 中的时间戳完整性；如果某段时间戳缺失，对应 phase 会直接不显示。
- 如果 HTML 在 IDE 里有静态报错，但浏览器能正常展示，通常只是编辑器对内嵌 JSON / script 的误报，不影响运行。
