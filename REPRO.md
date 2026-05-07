# 256 样本实验复现（首版）

更新时间：2026-05-07

## 1. 先决条件

- 已安装并可用 vllm,vllm-omni。
- 代码目录：/workspace/hw_verl/vllm-omni-adapter
- 外部依赖目录（默认同层）：
  - ../alpamayo1.5/src
  - ../verl-liming/my_example/alpamayo/src

## 2. 必要环境变量

建议在启动前设置：

```bash
export MODEL_PATH=/workspace/model/Alpamayo-1.5-10B
export DATASET_PATH=/workspace/dataset/Alpamayo_pai_av_big
export PROFILER_DIR=./profiler
```

说明：
- MODEL_PATH：模型目录。
- DATASET_PATH：本地数据集目录。
- PROFILER_DIR：torch profiler 输出目录（建议保持相对路径）。

## 3. 启动与运行

在仓库根目录执行：

```bash
cd /workspace/hw_verl/vllm-omni-adapter
source /workspace/.venv/bin/activate
python profiler/batch_sweep_cont_new.py
```

输出文件：
- profiler/batch_sweep_results_new.md

## 4. 请求上限与样本完整性

- 每组请求上限：<= 24（脚本中 MAX_REQ_PER_GROUP 控制）。
- 总样本数：256（含残组处理）。

## 5. 新增文件功能说明

### profiler/batch_sweep_cont_new.py

主实验入口。流程：
1. 依次遍历 `BS_LIST = [1, 2, 4, 8, 12, 16, 24]`，每个 BS 重启一次 vllm-omni 服务。
2. 将 256 个样本（22 个 unique clip 循环复用）按 `MAX_REQ_PER_GROUP = 24` 上限分批发给服务。
3. 每批通过 `ThreadPoolExecutor` 并发发送，收集 `custom_output` 字段里的逐请求 timing 指标。
4. 汇总生成三张表写入 `profiler/batch_sweep_results_new.md`。

关键常量：

| 常量 | 值 | 说明 |
|---|---|---|
| `N_TOTAL` | 256 | 总样本数 |
| `N_UNIQUE` | 22 | 唯一 clip 数（循环使用） |
| `BS_LIST` | [1,2,4,8,12,16,24] | 批大小列表 |
| `MAX_REQ_PER_GROUP` | 24 | 每次投递到 vLLM 的请求上限（KV cache 管理约束） |
| `PORT` | 8300 | 服务监听端口 |

---

### profiler/run_rollout_timing.py

数据加载与请求构建支撑库，被 `batch_sweep_cont_new.py` import。核心功能：

- `_load_local_avdi()`：初始化本地数据集接口（`PhysicalAIAVDatasetLocalInterface`），读取 `DATASET_PATH`，加载 `CHUNK_IDS = [3116]`。
- `_load_clip_data(clip_id, t0_us, avdi)`：按 clip_id + 时间戳加载视频帧、ego 轨迹信息。
- `_build_prompt_messages(camera_indices, num_frames_per_camera)`：构建包含轨迹占位符和摄像头标注的 system/user/assistant 消息结构。
- `_adapt_messages_with_frames(messages, frames)`：将 tensor 图像帧转换为 base64 data_url 并插入消息 content。
- `prepare_sample_request(clip_id, t0_us, avdi)`：一步接口，返回 `(request_messages, additional_information)` 供 HTTP 请求直接使用。

路径来源（均支持环境变量覆盖）：

| 变量 | 默认（环境变量未设置时） | 环境变量名 |
|---|---|---|
| `DATASET_PATH` | `/share/datasets/Alpamayo_pai_av_big` | `DATASET_PATH` |
| `MODEL_PATH` | `/share/models/Alpamayo-1.5-10B` | `MODEL_PATH` |

---

### profiler/alpamayo1_5_gpu0.yaml

vllm-omni 双阶段服务配置文件（单 GPU 部署，GPU 0）：

- **stage 0（VLM Reasoner）**：加载 `Alpamayo1_5Qwen3VLForConditionalGeneration`，使用 `OmniARScheduler`，开启 KV transfer（触发条件：`token_id: 151643`），输出 KV cache 给 stage 1。
- **stage 1（Trajectory Head）**：接收 stage 0 的 KV cache，运行扩散模型生成轨迹，`recv_timeout: 120.0`。

路径注入口（需对应设置环境变量）：

| 字段 | 环境变量 | 说明 |
|---|---|---|
| `model_subdir` | `MODEL_PATH` | VLM 模型目录 |
| `torch_profiler_dir` | `PROFILER_DIR` | torch profiler 输出目录，默认 `./profiler` |
| `tokenizer_subdir` | 无（固定 `/share`） | tokenizer 位于 `/share`，不需要改 |

---

### sitecustomize.py

Python 解释器启动时自动执行的钩子（只要仓库根目录在 `PYTHONPATH` 中即生效）。

作用：把 `alpamayo_reasoning_vla` 配置类型注册到 `transformers.AutoConfig`，使 vLLM 在加载 Alpamayo 模型时不报"未知 architecture"错误。无需手动调用，已有注册则静默跳过。

## 6. 常见问题

1) 服务未就绪
- 现象：health 检查超时。
- 处理：检查 profiler/logs/svc_bs*.log。

2) 模型路径错误
- 现象：启动时找不到模型。
- 处理：确认 MODEL_PATH 指向实际模型目录。

3) 数据路径错误
- 现象：加载 clip 失败。
- 处理：确认 DATASET_PATH 正确且权限可读。
