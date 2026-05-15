# Alpamayo Online Test

这个目录用于通过 `vllm-omni serve --omni` 启动 Alpamayo 服务，并用 OpenAI Chat Completions 接口发起在线请求。

**文件说明**
- [start_alpamoya_service.sh](/workspace/project/RL-learning/vllm-omni/tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_service.sh:1)：启动服务。
- [request_alpamoya_service.sh](/workspace/project/RL-learning/vllm-omni/tests/diffusion/models/alpamoya/custom_test/online/request_alpamoya_service.sh:1)：发送一条在线请求。
- [alpamoya_openai_client.py](/workspace/project/RL-learning/vllm-omni/tests/diffusion/models/alpamoya/custom_test/online/alpamoya_openai_client.py:1)：请求客户端，会解析 `metrics.custom_output`。
- [start_alpamoya_profile.sh](/workspace/project/RL-learning/vllm-omni/tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_profile.sh:1)：调用 `/start_profile`。
- [stop_alpamoya_profile.sh](/workspace/project/RL-learning/vllm-omni/tests/diffusion/models/alpamoya/custom_test/online/stop_alpamoya_profile.sh:1)：调用 `/stop_profile`。

**快速开始**
先启动服务：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_service.sh
```

再发送请求：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/request_alpamoya_service.sh
```

默认请求会打印：
- 请求耗时
- `pred_xyz` / `pred_rot` shape
- `cot_token_ids` shape
- 解析后的 `cot_text`
- 简单轨迹指标，例如 `minADE_m`

**常用环境变量**
- `HOST`：服务地址，默认 `127.0.0.1`
- `PORT`：服务端口，默认 `8000`
- `MODEL`：服务端模型名或路径
- `USE_HELPER_MESSAGES`：是否使用 helper 构造消息，默认 `false`
- `INCLUDE_RAW_RESPONSE`：是否在 client 输出里带完整原始响应，默认 `false`

请求脚本也支持直接传：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/request_alpamoya_service.sh <clip_id> <t0_us>
```

**采集服务日志与阶段指标**
启动时加 `--collect-metrics` 会打开：
- `--log-stats`
- `--enable-diffusion-pipeline-profiler`

示例：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_service.sh \
  --collect-metrics
```

如果想实时写日志文件，同时在终端看到输出：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_service.sh \
  --collect-metrics \
  --log-file /workspace/project/RL-learning/vllm-omni/learning/test/alpamayo_serve.log
```

说明：
- `--log-file` 是脚本侧用 `tee` 落盘，行为是实时的。
- `trans_table` 目前在 Alpamayo `stage0 -> stage1` 这条 `llm -> diffusion` 路径上通常是 0，这是当前统计实现覆盖不足，不代表没有实际数据传递。

**Torch Profiler**
当前 `stage1` 的 `profiler_config` 已经配在：
[alpamayo1_5.yaml](/workspace/project/RL-learning/vllm-omni/vllm_omni/model_executor/stage_configs/alpamayo1_5.yaml:68)

trace 默认输出目录：

```text
/workspace/project/RL-learning/vllm-omni/learning/test/torch_traces
```

使用方式：

1. 重启服务，让最新 stage config 生效。
2. 启动 profiler，默认只打 `stage1`：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_profile.sh
```

3. 发送一条请求：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/request_alpamoya_service.sh
```

4. 停止 profiler：

```bash
bash tests/diffusion/models/alpamoya/custom_test/online/stop_alpamoya_profile.sh
```

如果想 profile 全部 stage：

```bash
STAGES_JSON=all bash tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_profile.sh
STAGES_JSON=all bash tests/diffusion/models/alpamoya/custom_test/online/stop_alpamoya_profile.sh
```

如果想指定多个 stage：

```bash
STAGES_JSON='[0,1]' bash tests/diffusion/models/alpamoya/custom_test/online/start_alpamoya_profile.sh
STAGES_JSON='[0,1]' bash tests/diffusion/models/alpamoya/custom_test/online/stop_alpamoya_profile.sh
```

**关于 stage1 慢一些这件事**
这是正常现象。`stage1` 不是生成几个 token，而是在做 trajectory rollout，默认 `num_inference_steps=10`，所以通常会比 `stage0` 的 17 个输出 token 更慢。

**排查建议**
- 想看最终结构化结果：把 `INCLUDE_RAW_RESPONSE=true` 打开。
- 想看阶段耗时：用 `--collect-metrics`。
- 想看更细的算子级 trace：用 `start_alpamoya_profile.sh` / `stop_alpamoya_profile.sh`。
- 改了 `alpamayo1_5.yaml` 后记得重启服务。
