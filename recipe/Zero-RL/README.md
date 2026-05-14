# Zero-RL: DAPO vs GRPO 对比实验

Qwen3-4B-Base 上对比 DAPO（Decoupled Clip）和标准 GRPO（对称裁剪）。

## 实验设计

| 项目 | DAPO | GRPO |
|------|------|------|
| 模型 | Qwen3-4B-Base | 同左 |
| 训练数据 | dapo-math-17k（boxed prompt） | 同左 |
| 测试集 | MATH-500 + AIME 2024/2025 | 同左 |
| Clip | low=0.2, high=0.28 | 0.2（对称） |
| Overlong buffer | 是 | 否 |
| 答案验证 | math_verify（等价比较） | 同左 |
| Rollout | 16/题 | 同左 |
| Batch | train=512, mini=32 | 同左 |
| 学习率 | 1e-6 | 同左 |
| 步数 | 200 | 同左 |

## 用法

```bash
conda activate verl
cd /data3/yyy/verl

# 并行跑两个实验（各占 4 卡）
bash recipe/Zero-RL/run_dapo.sh &
bash recipe/Zero-RL/run_grpo.sh &
```
