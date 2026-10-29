# DSA resident-capacity trace: full-model TP8

该目录保存可审计的服务启动与数据采集入口。后续容量观测的启动和请求采集统一使用此目录中的版本，不再临时拼接命令。

## 文件

- `run_dsa_capacity_full_tp8.sh`：完整 GLM-5.3、TP=8 服务启动入口；负责环境检查、trace 目录创建和启动参数。
- `run_dsa_capacity_trace.sh`：设置只读观测环境变量并启动指定命令。
- `request_dsa_capacity_trace.py`：请求指定输入/输出 token 数，记录客户端请求与响应。
- 离线分析器仍使用仓库中的 `benchmarks/dsa/analyze_dsa_resident_capacity.py`，避免复制出第二份分析逻辑。

## 启动服务

在容器中从仓库根目录执行：

    bash benchmarks/dsa/capacity_trace/run_dsa_capacity_full_tp8.sh

如需后台启动：

    nohup bash benchmarks/dsa/capacity_trace/run_dsa_capacity_full_tp8.sh > /tmp/dsa-capacity-full-tp8.launcher.log 2>&1 < /dev/null &

服务启动后可通过 `/tmp/dsa_capacity_full_tp8_trace_dir` 读取本轮 TRACE_DIR；完整服务日志保存在该目录的 `server.log`。

## 发送采集请求

确认服务健康后，从仓库根目录执行：

    TRACE_DIR=$(cat /tmp/dsa_capacity_full_tp8_trace_dir)
    /usr/local/python3.12.13/bin/python benchmarks/dsa/capacity_trace/request_dsa_capacity_trace.py --base-url http://127.0.0.1:8080 --model glm --input-tokens 12000 --output-tokens 64 --seed 1024 --timeout 600 --trace-dir "$TRACE_DIR"

请求脚本先调用 `/tokenize` 生成精确 token ID，再请求 `/v1/completions`；请求和响应元数据写入 TRACE_DIR。

## 离线分析

    TRACE_DIR=$(cat /tmp/dsa_capacity_full_tp8_trace_dir)
    /usr/local/python3.12.13/bin/python benchmarks/dsa/analyze_dsa_resident_capacity.py --trace-dir "$TRACE_DIR" --capacities 0,1024,2048,4096,6144,8192,10240,12288

观测模式会将 TopK 数据复制到 CPU 并写入 JSONL，因此不得用该轮服务延迟作为性能基线。分析输出是 trace 驱动的 LRU 容量估计，不等同于真实运行时 HBM 命中率或 offload miss 实测。

## 配置基线与安全边界

- 本次启动脚本沿用已提供的 `run_dsa_capacity_full_tp8.sh` 配置，包括 `TASK_QUEUE_ENABLE=0`，不自行添加 `--async-scheduling`。
- `HCCL_IF_IP` 保持未设置；脚本中 `TEMP_DISABLED` 行是注释，不会设置该环境变量。
- 使用完整模型、TP=8、`--enforce-eager`，且只采集观测数据，不用于性能对比。
- 脚本在启动前检查模型配置、观测 hook、Python 导入路径和端口占用；检查失败即停止。
- 每轮创建独立 TRACE_DIR，并保存 `server.log`、NPU plog、manifest、TopK JSONL、client request/response 与分析输出。
- 如需更改启动参数，应修改并提交此目录下的脚本，保留 Git commit 作为历史审计记录。
