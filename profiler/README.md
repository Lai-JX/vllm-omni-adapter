# profiler

这个目录放的是和 Alpamayo / AsyncOmni profiling、batch sweep、时间线可视化有关的脚本与产物。

它大致分成 4 类：

1. 原始 sweep / 压测脚本
2. stage0 / two-stage 专项 profiling 脚本
3. 日志时间戳提取与 HTML 时间线渲染脚本
4. 已跑出的 logs / metrics / traces

## 目录概览

### 1. 通用或基础 sweep 脚本

- [batch_sweep_cont_inproc_new.py](batch_sweep_cont_inproc_new.py)
- [batch_sweep_cont_new.py](batch_sweep_cont_new.py)
- [batch_sweep_cont_inproc_tokenized.py](batch_sweep_cont_inproc_tokenized.py)
- [batch_sweep_cont_inproc_tokenized_warmup.py](batch_sweep_cont_inproc_tokenized_warmup.py)

其中最常作为后续 profiling 基础的是：

- [batch_sweep_cont_inproc_tokenized_warmup.py](batch_sweep_cont_inproc_tokenized_warmup.py)
  - 使用 tokenized Alpamayo 输入跑 in-process AsyncOmni。
  - 会生成 service log、torch trace、metrics markdown。
  - 提供很多被复用的辅助函数，例如：
    - 构造 tokenized 样本
    - 构造 runtime stage config
    - 重定向服务日志
    - 启动 / 清理 runtime

## 2. stage0 profiling

这组脚本关注 stage0 LLM 本身，不经过完整 two-stage diffusion 链路。

- [sweep_stage0_profile_vs_tokens.py](sweep_stage0_profile_vs_tokens.py)
  - 按 `sample_count` 合并自然样本，采集 stage0 profile。
  - 主要输出：
    - `embed_multimodal_ms`
    - `forward_ms`
    - 以及相关 token / cache 统计
- [plot_stage0_profile_vs_tokens.py](plot_stage0_profile_vs_tokens.py)
  - 从 stage0 metrics JSON 画静态图。
  - 当前包含：
    - `image_count vs embed_multimodal_ms`
    - `batch_total_scheduled_tokens vs forward_ms`
    - `embed_multimodal_ms / forward_ms` ratio 图

如果你只想看 stage0 编码与 forward 的缩放行为，通常从这组脚本开始。

## 3. two-stage profiling

这组脚本关注完整链路：

```text
stage0 LLM -> subsequent decode gap -> KV transfer -> stage1 diffusion
```

位于子目录：

- [two_stage_profiler/](two_stage_profiler/)

核心脚本：

- [two_stage_profiler/sweep_two_stage_profile_vs_tokens.py](two_stage_profiler/sweep_two_stage_profile_vs_tokens.py)
  - 主采集脚本。
  - 按 `sample_count` 构造 merged request，跑完整 two-stage pipeline。
  - 输出 JSON / CSV / service log / torch trace。
- [two_stage_profiler/plot_two_stage_profile_vs_tokens.py](two_stage_profiler/plot_two_stage_profile_vs_tokens.py)
  - 从 metrics JSON 生成 5 张静态图。
- [two_stage_profiler/render_two_stage_timestamp_timeline.py](two_stage_profiler/render_two_stage_timestamp_timeline.py)
  - 从 metrics JSON 生成交互式 HTML 时间线。

更详细说明见：

- [two_stage_profiler/README.md](two_stage_profiler/README.md)

## 4. 时间戳提取与时间线渲染

这组脚本面向 service log，适合做 request 级 phase timeline 可视化。

位于子目录：

- [timestamps/](timestamps/)

核心脚本：

- [timestamps/extract_request_timestamps.py](timestamps/extract_request_timestamps.py)
  - 从 service log 中解析 request 级 phase 时间戳。
- [timestamps/render_request_timeline.py](timestamps/render_request_timeline.py)
  - 渲染交互式 HTML timeline。
  - 更适合做 request-phase 总览，而不是直接看结构化 metrics JSON。

适用场景：

- 想从 service log 直接回看某次运行的时序
- 想定位某个 request 在 stage0 / KV / diffusion 上卡在哪里
- 想用交互式时间线检查批次内请求重叠关系

## 5. 其他脚本

- [run_rollout_timing.py](run_rollout_timing.py)
  - 模拟 GRPO rollout timing。
  - 通过 HTTP 请求服务，统计 rollout 场景下的 batch / request 时间。
- [regenerate_batch_results_md.py](regenerate_batch_results_md.py)
  - 用于从已有结果重新整理 markdown 结果。
- [alpamayo1_5_gpu0.yaml](alpamayo1_5_gpu0.yaml)
  - profiling / runtime 常用的 stage 配置来源文件。

## logs 与输出产物

### [logs/](logs/)

这个目录存放历史 profiling / batch sweep 运行结果。

常见内容包括：

- `svc_logs/`
  - 服务日志
- `metrics/`
  - 结构化结果、markdown 汇总、CSV
- `torch_traces/`
  - torch profiler traces
- `async_omni_inproc_bs*.log`
  - in-process 运行日志
- `svc_bs*.log`
  - 服务日志快照

### [two_stage_profiler/log/](two_stage_profiler/log/)

这个目录存放 two-stage profiling 的单独运行结果，每次 run 一般包含：

- runtime yaml
- metrics JSON / CSV
- engine metrics CSV
- service log
- torch traces
- timeline HTML

## 常用工作流

### 场景 1：只看 stage0 profile

1. 运行 [sweep_stage0_profile_vs_tokens.py](sweep_stage0_profile_vs_tokens.py)
2. 产出 `stage0_profile_vs_tokens.json`
3. 用 [plot_stage0_profile_vs_tokens.py](plot_stage0_profile_vs_tokens.py) 画图

### 场景 2：看完整 two-stage 链路

1. 运行 [two_stage_profiler/sweep_two_stage_profile_vs_tokens.py](two_stage_profiler/sweep_two_stage_profile_vs_tokens.py)
2. 查看 `two_stage_profile_vs_tokens_metrics.json`
3. 用 [two_stage_profiler/plot_two_stage_profile_vs_tokens.py](two_stage_profiler/plot_two_stage_profile_vs_tokens.py) 画静态图
4. 用 [two_stage_profiler/render_two_stage_timestamp_timeline.py](two_stage_profiler/render_two_stage_timestamp_timeline.py) 生成 timeline HTML

### 场景 3：直接从 service log 回看 phase timeline

1. 准备某次运行的 service log
2. 用 [timestamps/render_request_timeline.py](timestamps/render_request_timeline.py) 生成 HTML
3. 在浏览器中查看请求级 phase 分布

## 你通常应该先看哪个脚本

如果目标是：

- **看基础 tokenized sweep 如何跑起来**
  - 先看 [batch_sweep_cont_inproc_tokenized_warmup.py](batch_sweep_cont_inproc_tokenized_warmup.py)
- **只分析 stage0**
  - 先看 [sweep_stage0_profile_vs_tokens.py](sweep_stage0_profile_vs_tokens.py)
- **分析完整 two-stage timing**
  - 先看 [two_stage_profiler/sweep_two_stage_profile_vs_tokens.py](two_stage_profiler/sweep_two_stage_profile_vs_tokens.py)
- **分析日志级 phase 时间线**
  - 先看 [timestamps/extract_request_timestamps.py](timestamps/extract_request_timestamps.py)
  - 再看 [timestamps/render_request_timeline.py](timestamps/render_request_timeline.py)

## 备注

- 这个目录里的很多脚本依赖本仓库外部的数据集、本地模型路径和 Alpamayo 相关源码路径。
- 很多 sweep 行为通过环境变量控制，具体默认值直接看各脚本顶部常量定义最准确。
- 两条最常用的分析线：
  - **结构化 metrics 线**：适合画图、聚合统计
  - **service log timeline 线**：适合还原 request 时序与排队/phase gap
