# entropy_vis — 多模型逐 token 熵对比可视化

对若干基座/指令模型在数学题上 greedy 生成，记录每个生成位置的 top-K logprobs，前端用 HTML 网页交互展示熵分布、token 概率和跨模型对比。

## 文件

- `eval_entropy_all_models.py` —— vLLM 推理 + 熵计算（top-K 近似），输出 `entropy_all_models.json`。
- `entropy_vis.html` —— 前端可视化（单文件，零依赖，浏览器直开）。
- `run_eval.sh` —— 启动脚本。

## 用法

前置条件：先 `conda activate verl`（CUDA 12.8 toolchain 和 vLLM 都在该 env 内；脚本会强制检查，未激活会 fail）。

```bash
conda activate verl
cd recipe/entropy_vis

# 跑全部 7 个模型（默认 4 题，max_new=8192）
bash run_eval.sh

# 只跑指定模型
MODELS="Qwen2.5-3B-Instruct Qwen3-4B-Base" bash run_eval.sh

# 改题量 / 生成上限
N_PROBLEMS=8 MAX_NEW=4096 bash run_eval.sh
```

跑完后用任意静态服务器打开 html：

```bash
python -m http.server 8000   # 然后浏览器开 http://localhost:8000/entropy_vis.html
```

## 前端功能

页面分两种视图，右上角 **概览 / 详情** 切换（快捷键 `v`）。

### 详情视图（默认）

- **模型 tab**：badge 显示该模型在所有题上的正确数；hover 看 top-K / vocab。
- **题 tab**：`✓ / ✗` 标记当前模型在该题上是否答对。
- **左栏**
  - 题目（默认折叠，**展开**按钮全文显示），答案，verdict
  - **复制 response** / **查看完整 response** 折叠区
  - 统计行（token 数、熵均值/最大值、高熵占比）
  - **熵分布直方图**（固定档位 `0–0.5 / 0.5–1 / 1–2 / 2–3 / 3–5 / 5–10 / 10+`）
  - **熵滚动均值曲线**（window 随 token 数自适应）
  - **高熵 token Top-15 列表**，点击跳到对应位置
  - Token 流（按熵着色），点击查看详情
- **右栏（点 token 后）**
  - Top-K 候选概率 bar（实际生成的 token **高亮黄色**）
  - **同 token-idx 跨模型对比**：每个模型在第 N 个 token 位置的熵和 actual token
  - **同字符位置跨模型对比**：基于 `cum_char` offset 二分查找最接近的 token；不同 tokenizer 也能大致字符对齐（点击单元格会跳到该模型的对应 token）
- **可拖动分隔条**：左右栏宽度可拖动调整。
- **底部熵-位置曲线**：当前题，所有模型熵随 token 索引叠加；当前模型加粗加亮，竖线标出当前选中的 token。点曲线区域可跳到对应 token，legend 可单独开关每个模型。

### 概览视图

- **模型 × 题 矩阵**：单元格显示熵均值（按熵值着色）+ token 数 + 对错；最后一列/行是每行/列的均值。点击进入详情。
- **平均熵柱状图**：每个模型在所有题合并的平均熵。
- **正确率柱状图**：每个模型 `正确题数 / 总题数`。

### 键盘快捷键

| 键 | 作用 |
|---|---|
| `←` / `→` | 上一个 / 下一个 token |
| `[` / `]` | 上一题 / 下一题 |
| `1` – `9` | 切换到第 N 个模型 |
| `v` | 切换概览 / 详情视图 |

## 数据 schema

`entropy_all_models.json` 的结构（向后兼容旧数据，新增字段是 `cum_char`）：

```jsonc
{
  "problems":[{"idx":0,"question":"...","ground_truth":"..."}, ...],
  "models":[{
    "name":"Qwen2.5-3B-Instruct",
    "top_k_used":20,
    "vocab_size":151936,
    "problems":[{
      "idx":0, "ground_truth":"...", "response_text":"...",
      "response_len":2345, "correct":true, "extracted_answer":"...",
      "tokens":[{
        "pos":0, "token":"\nFirst",
        "entropy":0.123, "actual_prob":0.95,
        "cum_char":0,                    // ← 新增：起始字符 offset
        "topk":[{"token":"...","prob":0.95}, ...]
      }, ...]
    }, ...]
  }, ...],
  "meta":{"approx_entropy":true,"top_k":20,"max_new":8192,"n_problems":4}
}
```

旧的 JSON 缺 `cum_char` 也能正常加载；只是"同字符位置跨模型对比"那一栏会显示"无 offset"。要启用该对比，需要用最新版的 `eval_entropy_all_models.py` 重新生成。

## 关于熵的近似

vLLM 不返回完整 vocab 分布，只能取 top-K（最多 K=20）的 logprob。本脚本用以下公式近似熵：

```
H ≈ -Σ_{i∈topK} p_i log p_i  +  r·(log(V-K) - log r),  r = 1 - Σ_topK p_i
```

剩余质量 r 假设均匀分布在剩余词表上（这是熵的上界估计）。在 top-K 已覆盖大部分概率质量的低-中熵场景下，近似与真实熵差距 < 0.05 nats；高熵场景会偏高一点点，可接受。

如果需要精确熵，请加 HF 全 forward 路径（成本：每模型多一倍显存/时间）。

## 中断续跑

脚本自带断点续跑：若 `entropy_all_models.json` 已存在且 problems 一致，会自动跳过已完成模型。要重跑请删除 json 文件。
