# CLAUDE.md

## 项目概览

这是 [veRL](https://github.com/volcengine/verl)（v0.4.1）的 fork —— 字节跳动 Seed 团队开源的 LLM 强化学习训练框架，基于 Ray + PyTorch FSDP/Megatron-LM + vLLM/SGLang。

- `**verl/**`：官方上游实现。**尽量不修改这个目录下的文件**。
- `**recipe/`**：所有自定义实验和 recipe 放这里。
- `**examples/`**：官方启动脚本和实验配置。

## 关键路径

- **模型**：`/data3/yyy/models/`
  - Qwen2.5-3B-Instruct、Qwen2.5-7B
  - Qwen3-4B-Base、Qwen3-4B-Instruct-2507、Qwen3-8B-Base
- **数据**：`/data3/yyy/verl/data/`
- **主要自定义 recipe**：`recipe/RLSD/`（核心），以及 `recipe/train_gsm8k/`、`recipe/math_challenger_solver_judge/` 等

## Python / Conda 环境

预配置的 conda 环境：

- `verl` —— 主力 RL 训练环境

激活：`conda activate verl`。**所有训练/推理脚本都必须先激活**（不要用 `conda run -n verl` 或直接调 `/data3/yyy/miniconda3/envs/verl/bin/python`，会绕过下面的 activate hook）。

### CUDA 工具链

`verl` 环境里装了完整的 CUDA 12.8 toolkit，以匹配 torch `2.8.0+cu128`：

- conda 包：`cuda-nvcc` / `cuda-cudart-dev` / `cuda-crt` / `libcurand-dev` / `libcublas-dev`（channel `nvidia/label/cuda-12.8.0`）
- `activate.d/cuda.sh`：激活时设 `CUDA_HOME=$CONDA_PREFIX`，把 `$CONDA_PREFIX/bin` prepend 到 PATH
- `deactivate.d/cuda.sh`：退出时还原

flashinfer / torch cpp_extension 的 JIT 编译都会走这套环境的 nvcc 12.8。不要改这两个 hook 脚本，也不要在 recipe 启动脚本里重复导出 `CUDA_HOME` / `PATH`——只要正确激活 env 就够了。

如果以后要加新的 CUDA 头文件依赖（例如某个新 JIT 模块要 `cudnn.h`），走同样的 channel 装 `-dev` 包即可。

## 上游架构（`verl/`）

- `verl/single_controller/` —— 基于 Ray 的分布式编排层（WorkerGroup、ResourcePool、dispatch/execute 装饰器）
- `verl/protocol.py` —— `DataProto`：统一的组件间数据传输格式，封装了 TensorDict + dict + meta_info
- `verl/workers/` —— GPU 计算单元：
  - `actor/dp_actor.py` —— PPO 策略更新（loss 计算、梯度累积、micro-batching）
  - `rollout/` —— 通过 vLLM/SGLang 做推理生成
  - `critic/` —— 价值网络训练
  - `reward_manager/` —— 奖励计算（基于注册机制，可扩展）
- `verl/trainer/ppo/ray_trainer.py` —— `RayPPOTrainer`：RL 训练主循环
- `verl/trainer/ppo/core_algos.py` —— 优势估计（GAE、GRPO、REINFORCE++、RLOO 等）和策略损失函数
- `verl/trainer/config/` —— Hydra/OmegaConf YAML 配置（ppo_trainer.yaml、ppo_megatron_trainer.yaml、sft_trainer.yaml）
- `verl/models/` —— 模型注册（transformers：Llama、Qwen2/VL；mcore：Megatron）

支持的 RL 算法：PPO、GRPO、DAPO、RLOO、REINFORCE++、ReMax、PRIME、PF-PPO、VAPO、DrGRPO、KL_Cov、Clip_Cov

## 自定义 Recipe（`recipe/`）

所有自定义实验都放在 `recipe/<exp_name>/` 下，一实验一目录，不动 `verl/` 上游。新起实验时也遵循此约定。

目前已有的 recipe：

| 目录 | 性质 | 说明 |
|------|------|------|
| `recipe/char_count/` | 上游教学示例 | RLVR 入门：字符计数任务，135M 小模型 + 8GB GPU 可跑 |
| `recipe/dapo/` | 上游算法 | DAPO（Decoupled Clip and Dynamic Sampling PO）实现与复现脚本 |
| `recipe/few_data_thinking/` | 自建 | GRPO 少量数据训练实验（5 道数学题起步） |
| `recipe/math_challenger_solver/` | 自建 | 数学对抗式训练：出题模型 A + 解题模型 B，伪标签由 B 多数投票产生 |
| `recipe/math_challenger_solver_judge/` | 自建 | 上一个 recipe 的 judge 变体，引入 judge 模型参与评分 |
| `recipe/entropy_vis/` | 自建工具 | 多模型逐 token 熵对比可视化（vLLM + HTML 前端） |
| `recipe/sd_vis/` | 自建工具 | Self-distillation teacher/student logit 诊断可视化 |
| `recipe/RLSD/` | 已终止（2026-05-09） | RL + Self-Distillation：解决 GRPO 组内全错零梯度问题。实验发现表面增益来自"记忆捷径"而非真实推理，详见 `recipe/RLSD/README.md` 末尾结论 |

每个 recipe 目录自带 README（或直接看代码），有具体用法、数据、实验结论。修改/新增 recipe 前先读对应 README，避免误伤已有实验产物。

## Git

- 远程：`origin` → `https://github.com/yuanyueyao/verl`（你的 fork）
- 当前分支：`main`
- 上游：字节跳动 `volcengine/verl`。**不要从上游拉取更新**，锁定当前 verl 版本，避免合并冲突

## Agent 运行长任务（tmux 约定）

任何耗时 > 5 分钟的训练 / 推理 / 评测（vLLM 生成、RL 训练、SFT、大批量数据处理等）**agent 自己起 tmux 跑**，不要让用户手动 `bash xxx.sh`。

### SOP

1. **会话命名**：`verl-<recipe>-<短描述>`，例：`verl-entropy_vis-eval`、`verl-rlsd-train-7b`、`verl-sd_vis-gen`。
2. **启动模板**（推荐：把命令写进一个 `.sh` 脚本，再 `tmux new -d` 执行；比内联引号嵌套更稳）：

   ```bash
   TS=$(date +%Y%m%d_%H%M%S); LOG=<dir>/_logs/${TS}_<purpose>.log
   mkdir -p <dir>/_logs
   cat > /tmp/_run_<session>.sh <<'SH'
   #!/usr/bin/env bash
   source /data3/yyy/miniconda3/etc/profile.d/conda.sh
   conda activate verl
   cd <dir>
   <cmd> 2>&1 | tee "<LOG_PATH>"
   echo "=== DONE exit=$? ==="
   exec bash
   SH
   sed -i "s|<LOG_PATH>|$LOG|" /tmp/_run_<session>.sh
   chmod +x /tmp/_run_<session>.sh
   tmux new -d -s <session> "/tmp/_run_<session>.sh"
   ```

   - **不要** 用一行内联（`tmux new -d -s X "...echo '=== DONE exit=\$? ==='..."`）——多层引号嵌套时 `$?` 会被转义错，完成 marker 拿不到正确 exit code。
   - `source conda.sh` 而不是依赖 `~/.bashrc`，更稳。
   - `tee` 到 `<recipe>/_logs/<timestamp>_<purpose>.log` —— 日志永久保留，便于事后排查。
   - 末尾 `exec bash` 保活，方便 attach 看；`=== DONE exit=N ===` 作为完成 marker，可用 `grep` 或 `tmux has-session` 检测。
3. **监控**：
   - 截屏：`tmux capture-pane -pt <session> -S -50`（看最近 50 行）
   - 看完整日志：`tail -n 200 -f <recipe>/_logs/<file>.log`
   - 完成检测：`grep -q '=== DONE exit=' <log>` 或 `tmux has-session -t <session>` 配合
   - **优先用 `Read`/`Grep` 工具读日志文件**，不要依赖 `tmux capture-pane` + Shell 的输出。
     原因：Cursor agent shell 在某些情况下（特别是 `AwaitShell` 被用户中断后）会进入"输出黑洞"——命令瞬间返回 exit=0 但 stdout 全空。日志落盘到文件后用 file-based tool 读最稳。
4. **结束 / 清理**：任务完成后用 `tmux kill-session -t <session>` 释放（除非用户要保留窗口）。
5. **GPU 选择**：开跑前先 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`，再决定 `CUDA_VISIBLE_DEVICES`，**不要无脑占满 8 卡**——服务器是共享的。
6. **不要用** `nohup` / Cursor 后台 shell（`block_until_ms:0`）跑长任务——前者关 shell 会受影响，后者跨会话不持久；都不如 tmux 可控。

### 单 GPU vLLM 评测的常用变体

```bash
CUDA_VISIBLE_DEVICES=0 bash run_eval.sh
```

需要多卡推理时再考虑 vLLM 的 `tensor-parallel-size`。

