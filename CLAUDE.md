# CLAUDE.md


---

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
