# RLSD 项目上下文记忆

> 写给新 agent / 新机器上的你。读完这篇 + 看一眼代码，就能理解全貌并开始工作。

---

## 一、项目目标

发 EMNLP 2026 论文（截稿 May 25 UTC-12）。

**核心 claim**：在 self-distillation (OPSD) 的 per-token KL loss 中 mask 掉 epistemic token 位置，可以防止数学推理退化。

---

## 二、背景论文（必读）

**《Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?》** (Kim et al., 2026)
→ `recipe/new-idea/_mineru/full.md`

关键发现：
1. SDPO (self-distillation) 在数学推理上导致 AIME24 退化 ~40%
2. 根因：teacher 拿到 GT 答案后，推理变得过度 confident，抑制了 epistemic verbalization
3. Epistemic tokens: {wait, hmm, perhaps, maybe, actually, alternatively, seems, might, likely, check}
4. Teacher-student KL divergence 在 epistemic tokens 上是普通 token 的 2-8 倍（Appendix B.6）

---

## 三、我们的方法

**Uncertainty-Aware Self-Distillation = per-token masking of epistemic markers**

在 SD 的 token-level KL loss 中：
- `mask[t] = 0` 当 student 生成的 token 是 epistemic token（"Wait", "Hmm" 等）
- `mask[t] = 1` 否则
- `loss = Σ(mask[t] * KL_t) / Σ(mask[t])`

即：epistemic token 位置不参与 distillation，保护不确定性表达。

---

## 四、实验结论（截至 May 14）

### 4.1 Masked OPSD (token-identity) — 主要结果 ✅

- 模型：DeepSeek-R1-Distill-Qwen-7B
- 数据：Openthoughts Math 30k
- 配置：opsd_only=true, sd_mask_mode=token_identity
- 200 steps，8×A800-80GB

| Step | AIME24 acc@12 | AIME25 acc@12 | MATH-500 pass@1 |
|------|---------------|---------------|-----------------|
| 0    | 0.478         | 0.381         | 0.868           |
| 10   | 0.522         | 0.356         | 0.876           |
| 50   | 0.522         | 0.353         | 0.884           |
| 100  | 0.475         | 0.358         | 0.878           |
| 150  | 0.478         | 0.372         | 0.884           |
| 170  | 0.481         | 0.347         | 0.890           |

**结论**：AIME24 完全平坦，零退化。mask 有效。

### 4.2 Naive OPSD (无 mask) — 正在跑 ⏳

- 配置：sd_mask_mode=none，其他完全一样
- tmux: `verl-naive-opsd`
- 日志：`/data3/yyy/verl/logs/rlsd/20260514_082451_naive_opsd.log`
- WandB: `naive-opsd-dsr1-7b-{timestamp}`

**问题**：到 step 10 为止，naive OPSD 没有表现出论文 SDPO 那样的退化（AIME24 step 0=0.478, step 10=0.486）。

**原因分析**：我们的 teacher 看到的是**外部 reference_solution**（Openthoughts 数据集自带），不是像论文 SDPO 那样看到"模型自己生成的正确答案"（regeneration prompt）。外部 solution 和模型风格有 gap，collapse 慢得多。

```
论文 SDPO: teacher 看 "Correct solution: {自己刚写的} 再解一次" → 正反馈 → 快速 collapse
我们的 OPSD: teacher 看 "Below is a verified reference solution... {外部解}" → 温和 → 慢
```

---

## 五、关键代码文件

```
recipe/RLSD/
├── run_exp_masked_sd_tokenid.sh     # 跑 masked OPSD
├── run_exp_naive_opsd.sh            # 跑 naive OPSD baseline
├── main_rlsd.py                     # 训练入口
├── config/rlsd_trainer.yaml         # Hydra 配置（含 rlsd.sd_mask_mode）
├── rlsd/
│   ├── epistemic_mask.py            # ★ 核心：epistemic token ID 映射 + mask 生成
│   ├── loss.py                      # ★ 核心：compute_sd_loss_chunked (支持 token_mask)
│   ├── rlsd_actor.py                # ★ 核心：_update_sd() 注入 mask
│   ├── rlsd_trainer.py              # 训练循环 + SD/GRPO 路由
│   ├── prompt.py                    # Teacher/Student prompt 构造
│   ├── dataset.py                   # RLSDDataset + RLSDProblem
│   └── verifier.py                  # 答案验证（\boxed{} 提取 + math-verify）
```

### 数据流（SD 分支）

```
trainer._rlsd_step()
  → build_teacher_privileged_messages(question, ground_truth, reference_solution)
     → teacher prompt: "Problem: {q}\nBelow is a verified reference solution...\n{ref_sol}"
  → _build_sd_train_batch() → DataProto with student & teacher prompts + shared response
  → meta_info["sd_mask_mode"] = "token_identity" | "none" | "entropy_percentile"
  → meta_info["epistemic_token_ids"] = [id1, id2, ...]  (if token_identity)
  → actor_rollout_wg.update_actor()

actor._update_sd()
  → 读 sd_mask_mode from meta_info
  → 对于 token_identity: build_token_identity_mask(responses, epistemic_ids)
  → Teacher forward (no_grad, 特权 context) → ref_logits
  → Student forward (有梯度, 无特权) → stu_logits
  → compute_sd_loss_chunked(stu, ref, ..., token_mask=mask)
     → per-token KL × (response_mask × token_mask) / denom
  → loss.backward()
```

---

## 六、待做测试（在新机器上）

### 测试目的

验证 teacher 在有/无 reference_solution 作为特权上下文时，生成的响应是否有 epistemic token 差异。

### 测试步骤

```bash
conda activate verl
cd /data3/yyy/verl

# 1. 加载 DS-7B 模型
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/data3/yyy/models/DeepSeek-R1-Distill-Qwen-7B"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

# 2. 选一道题（从 Openthoughts 数据）
import pandas as pd
df = pd.read_parquet("/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet")
row = df.iloc[0]
question = row["problem"]
ref_solution = row.get("solution", "")
answer = row["Answer"]

# 3. 构造两种 prompt
# Student prompt（无特权）
student_prompt = f"Problem: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# Teacher prompt（带 reference_solution）
teacher_prompt = f"Problem: {question}\n\nBelow is a verified reference solution showing how the answer is derived. Use it to reason about the problem; your own response wording may differ.\n\n{ref_solution[:3000]}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# 4. 生成并统计 epistemic tokens
EPISTEMIC = {"wait", "hmm", "perhaps", "maybe", "actually", "alternatively", "seems", "might", "likely", "check"}

for name, prompt in [("Student (unguided)", student_prompt), ("Teacher (w/ ref sol)", teacher_prompt)]:
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(inputs, max_new_tokens=4096, temperature=0.6, do_sample=True)
    text = tok.decode(out[0], skip_special_tokens=True)
    # 统计
    words = text.lower().split()
    count = sum(1 for w in words if any(e in w for e in EPISTEMIC))
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"  Response length: {len(text)} chars")
    print(f"  Epistemic tokens: {count}")
    print(f"  Preview: {text[:500]}...")
EOF
```

### 期望结果

如果 hypothesis 正确：
- Student (unguided): 较多 epistemic tokens（~10-20+）
- Teacher (w/ ref sol): 较少 epistemic tokens（~3-8），回复更简洁

这能直接验证：**外部 reference_solution 也会抑制 epistemic verbalization，但程度可能比论文 SDPO 的 regeneration 弱。**

---

## 七、其他重要信息

### 代理

服务器上有 clash 代理运行在 `127.0.0.1:7890`：
```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```
当前选中香港节点，Google/GitHub/YouTube 都通。

### GPU 状态

- gpu-node13 (当前): 8×A800-80GB，全部被 naive OPSD 占用
- gpu-nodeXX (另一台): 待确认

### Paper LaTeX

`papers/emnlp2026/main.tex` — 草稿已完成，待填入 naive OPSD 数据。

### WandB

- Project: `rlsd`
- Masked OPSD: `masked-sd-tokenid-dsr1-7b-{ts}`
- Naive OPSD: `naive-opsd-dsr1-7b-{ts}`

---

*Generated: 2026-05-14. Update as needed.*
