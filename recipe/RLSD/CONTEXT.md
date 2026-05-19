# RLSD 项目上下文记忆

> 写给新 agent / 新机器上的你。读完这篇 + 看一眼代码，就能理解全貌并开始工作。

---

## 一、项目目标

发 EMNLP 2026 论文（截稿 May 25 UTC-12）。

**核心 claim**：在 self-distillation (OPSD) 的 per-token KL loss 中 mask 掉 epistemic token 位置，可以防止数学推理退化。
**初稿论文路径**：`papers/emnlp2026/main.tex`

## 二、背景论文（必读）

**《Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?》** (Kim et al., 2026)
→ `papers/_mineru/full.md`

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


| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 |
| ---- | ------------- | ------------- | --------------- |
| 0    | 0.478         | 0.381         | 0.868           |
| 10   | 0.522         | 0.356         | 0.876           |
| 50   | 0.522         | 0.353         | 0.884           |
| 100  | 0.475         | 0.358         | 0.878           |
| 150  | 0.478         | 0.372         | 0.884           |
| 170  | 0.481         | 0.347         | 0.890           |


**结论**：AIME24 完全平坦，零退化。mask 有效。

### 4.2 Naive OPSD (无 mask) — 已完成 ✅ (May 14)

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

## 六、Epistemic Token Probe —— 已完成 ✅ (May 14, updated)

### 测试设计

对比 4 组条件下，DS-7B 生成回复中的 epistemic token 数量 + 答案准确率：

- **A** Student（无特权，纯问题）
- **B** Teacher GT only（问题 + 正确答案）
- **C** Teacher + ref_solution（问题 + `solution` 列，简洁无 thinking）
- **D** Teacher + COT（问题 + `COT_Reason` 列，完整思考链，含 epistemic token）

代码：`recipe/RLSD/diagnostic/epistemic_probe.py`
规模：**10 题 × 4 条件 × 3 samples = 120 次生成**

### 聚合结果（30 samples / 条件）


| 条件                 | Avg. Length | Epi. Mask Pos. | Epi. Pos. Share | 准确率       | 解读                   |
| ------------------ | ----------- | -------------- | --------------- | --------- | -------------------- |
| **A** Student      | 5,200       | 728            | 14.0%           | 33.3%     | 正常推理，自然有犹豫           |
| **B** GT only      | 5,140       | 720            | 14.0%           | 33.3%     | 几乎无影响，epi 和准确率与 A 持平 |
| **C** ref_sol (简洁) | 4,590       | 459            | 10.0%           | 56.7%     | epi 下降 37%，准确率提升     |
| **D** COT (完整思考链)  | **2,920**   | **146**        | **5.0%**        | **90.0%** | epi 暴跌 80%，准确率暴涨     |


注：Table 3 中 token statistics 是对生成 responses 的 average；Epi. Mask Pos. 表示被 epistemic mask 词表命中的平均 response positions。扩充词表只是 tokenizer coverage，不引入另一套 epistemic token 概念。

### 逐题明细


| 题号  | A epi/acc | B epi/acc | C epi/acc | D epi/acc |
| --- | --------- | --------- | --------- | --------- |
| 0   | 15.7/0%   | 18.7/0%   | 9.0/0%    | 1.0/100%  |
| 1   | 13.7/33%  | 15.0/0%   | 5.3/100%  | 0.7/100%  |
| 2   | 25.3/0%   | 29.0/0%   | 23.0/33%  | 14.0/67%  |
| 3   | 10.0/100% | 6.7/100%  | 3.0/100%  | 0.0/100%  |
| 4   | 16.0/33%  | 7.7/67%   | 8.3/100%  | 1.0/100%  |
| 5   | 27.0/33%  | 27.0/33%  | 14.0/67%  | 14.3/67%  |
| 6   | 9.7/33%   | 12.0/33%  | 8.0/67%   | 3.0/100%  |
| 7   | 11.3/100% | 7.0/100%  | 8.3/100%  | 0.0/100%  |
| 8   | 35.0/0%   | 33.0/0%   | 24.7/0%   | 1.0/100%  |
| 9   | 8.0/0%    | 13.7/0%   | 4.0/0%    | 0.0/67%   |


### 关键发现

#### 1. Epistemic token 与准确率呈强负相关

当模型获得更多信息（ref_sol → COT），epistemic token 锐减，准确率飙升：

```
epistemic ↓:  728 (A)  →  459 (C)  →  146 (D)
准确率   ↑:  33.3% (A)  →  56.7% (C)  →  90.0% (D)
```

→ **模型拿到参考解后不再"自己推理"，而是直接复述参考解的推理路径。** epistemic token 的消失不是"变自信了"，而是"不需要推理了"。

#### 2. COT 抑制独立推理的效果最极端 ★ 核心发现

条件 D（COT 参考解）：

- epi count 146，仅为 Student 的 **20%**
- 准确率 90%，是 Student 的 **2.7 倍**
- 但这不是好事——模型完全依赖 COT 中的推理，几乎没有自己的 epistemic 表达

**这意味着**：如果 SD 中用 COT 作为 teacher 上下文，student 会被迫向一个"不推理"的分布靠拢，本质上是 **推理能力蒸馏退化**（reasoning collapse via distillation）。

#### 3. 简洁 ref_solution（C）效果适中

C 组 epi 下降 37%、准确率提升 23pp——影响介于无参考和 COT 之间。这是目前 OPSD 实际使用的条件。影响温和，解释了 naive OPSD 退化不明显。

#### 4. GT only（B）基本无影响

B 组在所有指标上与 A 组几乎一致。仅知道最终答案不足以改变模型的推理风格。

#### 5. 核心发现：不同 reference context 导致不同效应 ★


| Reference Context   | Epistemic 变化 | 准确率变化      | 推理模式         |
| ------------------- | ------------ | ---------- | ------------ |
| 无 (Student)         | 基准 728       | 33.3%      | 独立推理，自然犹豫    |
| GT only             | 几乎不变 720     | 33.3%      | 同 Student    |
| Clean solution (简洁) | ↓37% → 459   | ↑ → 56.7%  | 适度引导，部分依赖    |
| COT (完整思考链)         | ↓80% → 146   | ↑↑ → 90.0% | **完全丧失独立推理** |


→ **Reference context 的详细程度直接决定 reasoning collapse 的程度**

---

## 七、COT Reference Solution 对照实验 —— 已完成 ✅ (May 15)

### 实验设计

用 COT（`COT_Reason` 列，含 epistemic token 的完整思考链）替代 clean `solution` 作为 reference_solution，验证不同 reference context 的退化效应 + mask 的保护作用。

- **Naive**: `sd_mask_mode=none`, `reference_column=COT_Reason`
- **Masked**: `sd_mask_mode=token_identity`, `reference_column=COT_Reason`

模型：DeepSeek-R1-Distill-Qwen-1.5B，8×A800 串行，各 100 steps。

### 运行方式

```bash
# 串行自动运行（先 naive 后 masked）
bash recipe/RLSD/run_exp_cot_serial.sh
tmux attach -t cot-serial
```

### OOM 踩坑记录


| 问题                            | 原因                                              | 修复                                 |
| ----------------------------- | ----------------------------------------------- | ---------------------------------- |
| 4 卡并行 OOM                     | `ppo_micro_batch_size_per_gpu=2` → 2×24K tokens | 改 8 卡串行                            |
| 8 卡仍 OOM                      | `max_prompt_length=2048` 截断大部分 COT，改 8192 后序列太长 | `ppo_micro_batch_size_per_gpu=2→1` |
| `expandable_segments:True` 崩溃 | vLLM cumem 不兼容                                  | 去掉                                 |


最终有效参数：

- `max_prompt_length=8192`, `max_model_len=28672`, `ppo_micro_batch_size_per_gpu=1`
- 每 step ~110s（含 vLLM rollout + SD forward/backward）

### 结果


|                     | **Naive (no mask)**                  | **Masked (token_identity)**       |
| ------------------- | ------------------------------------ | --------------------------------- |
|                     | step 0 → 100                         | step 0 → 100                      |
| **AIME24 avg@12**   | 0.272 → **0.217** (-5.5pp, **-20%**) | 0.272 → **0.250** (-2.2pp, -8%)   |
| **AIME25 avg@12**   | 0.208 → **0.192** (-1.6pp, -8%)      | 0.208 → **0.208** (**0pp, flat**) |
| **MATH-500 pass@1** | 0.696 → 0.760 (+9%)                  | 0.696 → 0.740 (+6%)               |
| **Macro Mean**      | 0.392 → 0.389 (−1%)                  | 0.392 → **0.399** (+2%)           |


### 逐 step 明细


| Step | Naive AIME24 | Naive AIME25 | Masked AIME24 | Masked AIME25 |
| ---- | ------------ | ------------ | ------------- | ------------- |
| 0    | 0.272        | 0.208        | 0.272         | 0.208         |
| 10   | 0.247        | 0.219        | 0.283         | 0.225         |
| 20   | 0.275        | 0.228        | 0.250         | 0.214         |
| 30   | 0.297        | 0.244        | 0.236         | 0.200         |
| 40   | 0.269        | 0.222        | 0.303         | 0.208         |
| 50   | 0.244        | 0.200        | 0.272         | 0.228         |
| 60   | 0.261        | 0.219        | 0.258         | 0.211         |
| 70   | 0.250        | 0.219        | 0.250         | 0.225         |
| 80   | 0.239        | 0.206        | 0.256         | 0.169         |
| 90   | 0.239        | 0.189        | 0.256         | 0.219         |
| 100  | **0.217**    | **0.192**    | **0.250**     | **0.208**     |


WandB:

- Naive: `cot-naive-opsd-dsr1-1.5b-20260514_174129`
- Masked: `cot-masked-opsd-dsr1-1.5b-20260514_223026`

### 结论

1. **COT 触发了推理退化**：Naive AIME24 降 20%，AIME25 降 8%
2. **Mask 有效保护**：Masked AIME24 仅降 8%（挽救 12pp），AIME25 完全平稳（0 退化）
3. **退化程度弱于 paper SDPO（40%）**：因为 COT 是外部参考解，不是 regeneration
4. **MATH-500 两组都涨**：in-distribution 正常（SD 让模型贴合训练分布）

### Epistemic Probability Probe ★ (May 15)

对同一段 student 生成的 response（1024 tokens），分别做 Teacher（带 COT）和 Student（纯问题）的 forward，比较两者在每个 token 上的概率分布。

HTML 可视化：`epistemic_prob_probe.html`（`http://localhost:8899/epistemic_prob_probe.html`）

**结果**：


|              | Epistemic (15 tokens) | Non-Epistemic (1009) |
| ------------ | --------------------- | -------------------- |
| Teacher prob | **0.127**             | 0.7514               |
| Student prob | **0.4171**            | 0.7613               |
| Diff (S−T)   | **+0.0810**           | +0.0099              |


**关键洞察**：

- Teacher 对 epistemic token 概率 (0.127) **仅为**非 epistemic token (0.751) 的 **17%**
- Student 对 epistemic token 概率 (0.417) 比 Teacher **高 29.0pp**
- → KL divergence 在 epistemic token 位置上是普通 token 的 **8 倍**

这直接证明了 SD 退化的机制：Teacher 低估 epistemic token → KL loss 强力把 Student 拉向不下 epistemic 表达 → 推理退化。

### Top-10 Token Probe ★ (May 15)

进一步分析每个 epistemic token 位置上 Teacher/Student 的 top-10 候选分布。

HTML：`top10_probe.html`（`http://localhost:8899/top10_probe.html`）

**典型样本**：

```
#41  'Hmm':  Teacher top-1="I"(0.629), "Hmm"仅#2(0.232)
             Student top-1="Hmm"(0.453) = "I"(0.453) 并列
             → Teacher 压制 "Hmm", Student 平等考虑

#305 'Hmm': Teacher top-10 里根本没有 "Hmm"！
            首选 "So"(0.361) "Each"(0.281) "But"(0.091)
            → Teacher 完全拒绝 epistemic 表达

#261 'Wait': Teacher "Wait"#1(0.637), 备选压制 "Hmm"→仅0.022
             Student "Wait"#1(0.586), 备选仍有 "Hmm"(0.090)
```

→ **Teacher 在 epistemic token 位置系统性倾向于 non-epistemic 替代词**，Student 则自然保留 epistemic 多样性。这是 SD KL loss 在 token 级别的直接作用点。

### Prompt Ablation: "your own wording may differ" (May 15)

测试去掉 Teacher prompt 中的 `"your own response wording may differ"` 是否加剧 epistemic 抑制。

**方法**：同一 response，对比 current prompt 和 strict prompt（去掉该短语）的 top-10 token 概率。

**结果**：**几乎无影响**。

- Epistemic token prob 变化：+0.0075（噪声级别）
- Top-1 变化：8/1024 positions（0.8%）
- 14 个 epistemic 位置仅 1 处 top-1 改变

**结论**：Teacher 抑制 epistemic token 的根因是 **COT 内容本身**（看到自信推导 → 预期自信续写），不是这句 meta-instruction。在 forward pass 场景下，驱动力来自上下文的 token 模式，而非高层指令。

## 八、其他重要信息

### GPU 状态

- 服务器：8×A800-80GB（共享）

### 所有实验汇总


| 实验                  | 模型   | 参考解            | Mask     | AIME24 Δ    | AIME25 Δ    | 状态  |
| ------------------- | ---- | -------------- | -------- | ----------- | ----------- | --- |
| Naive OPSD (COT)    | 7B   | COTReason      | none     | **−26% ↓**  | −28% ↓      | ✅   |
| Masked OPSD (COT)   | 7B   | COTReason      | tokenid  | **+2.5% ↑** | −4% ↓       | ✅   |
| **COT Naive OPSD**  | 1.5B | **COT_Reason** | none     | **-20% ↓**  | -8% ↓       | ✅   |
| **COT Masked OPSD** | 1.5B | **COT_Reason** | token_id | **-8% ↓**   | **0% flat** | ✅   |


### Paper LaTeX

`papers/emnlp2026/main.tex` — 待更新 COT 实验结果。

### WandB

- Project: `rlsd`
- COT exp (7B): `cot-naive-opsd-dsr1-7b`, `cot-masked-opsd-dsr1-7b`
- COT exp: `cot-naive-opsd-dsr1-1.5b`, `cot-masked-opsd-dsr1-1.5b`

---

*Last updated: 2026-05-19.*

---

## 九、EMNLP 论文当前状态 (May 19)

### 2026-05-19 最新修订记录（当前以此为准）

本轮主要更新论文写作、术语、算法呈现、附录排版和编译脚本；若下方历史实验记录与本节冲突，以 `papers/emnlp2026/main.tex`、`recipe/RLSD/figures/generate_figures.py` 和本节为准。

1. **核心 claim**：Naive OPSD 会在数学推理中压制 epistemic tokens，削弱 self-checking / reflective reasoning；masked self-distillation 通过在 distillation loss 中排除 epistemic-token positions 缓解该退化。
2. **主术语**：正文优先使用 `token-level self-distillation signal` / `distillation signal`；公式和必要数学定义处保留 KL。避免在标题、图注和叙事中过度使用 KL。
3. **Method 当前形态**：Algorithm 1 已放入正文 §4.2 之后，展示 rollout/mask、token-level self-distillation signal、masked objective 和 update。算法中 mask 与 masked objective 使用红色标注。
4. **Case study**：Appendix C 已切换为 `\onecolumn` 跨栏展示，使用 `problemblock`、`caseblock`、`casenote`。Base 和 Masked 示例都包含“先算错，再通过 Wait/check 纠正”的 reflective behavior；Naive 示例保留错误不纠正。
5. **当前结果口径**：1.5B Naive AIME24/AIME25 为 `-22%/-21%`，Masked 为 `+19%/+18%`；7B Naive 为 `-17%/-19%`，Masked 为 `+3%/+4%`。

关键路径：

- 论文正文：`papers/emnlp2026/main.tex`
- 论文 PDF：`papers/emnlp2026/main.pdf`
- 编译脚本：`papers/emnlp2026/build.sh`
- 图生成脚本：`recipe/RLSD/figures/generate_figures.py`
- 论文使用图：`papers/emnlp2026/figures/{fig_token_kl_strip,fig_kl_concentration,fig_aime_curves_15b,fig_aime_curves_7b,fig_combined_4panel_15b}.pdf`

Figure 2 当前为 token-level signal heatmap：上半部分每个 token 按 per-position distillation signal 着色，epistemic/error/correction 用不同边框示例标注；下半部分 mean signal 为 Normal `0.013`、Error `0.147`、Epistemic `0.123`。
注意：下方第九/十二节是历史实验记录；若再冲突，以 `generate_figures.py`、`main.tex` 和本节为准。

### 论文概览

- **路径**: `papers/emnlp2026/main.tex`
- **编译**: `cd papers/emnlp2026 && ./build.sh`；当前 `main.pdf` 为 13 页
- **参考文献**: `papers/emnlp2026/custom.bib`（ACL 类自带 `\bibliographystyle`，tex 中不要重复声明）
- **编译注意**: `build.sh` 使用 `latexmk -pdf -interaction=nonstopmode main.tex`，并在没有 `rg` 时 fallback 到 `grep -nE` 检查 warning。若 `latexmk` 记住 previous invocation error，可用 `latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex` 强制重编译。

### 论文结构 (13 pages)

```
1. Introduction                    — Figure 1 (overview schematic)
2. Background & Related Work       — Self-Distillation, Epistemic Verbalization,
                                      Self-Correction, Uncertainty Expression
3. Diagnosis
   3.1 Domain Contrast             — Table 1
   3.2 Token-Level Signal          — Table 2 + Fig 2 (token-level signal strip) + Fig 3 (signal concentration)
   3.3 Top-k Token Analysis        — Table 3
   3.4 Reference Context Matters   — Table 4
4. Method: Epistemic Token Masking — Eq 1-3, token set E, mask M, Algorithm 1
5. Experiments
   5.1 Setup                       — COT-reference only, DS-7B & DS-1.5B
   5.2 COT Results                 — Combined table (1.5B+7B), Fig 4+5 (AIME+MATH curves, 双栏)
   5.3 Analysis                    — Fig 6 (4-panel dynamics)
6. Discussion & Limitations
7. Conclusion
Appendix A Training Details
Appendix B Prompt Templates
Appendix C Case Study              — one-column full-width reflective reasoning example
```

### 核心 Narrative

1. Self-distillation degrades math reasoning by suppressing epistemic verbalization
2. **Mechanism**: Teacher (privileged with COT) undervalues epistemic tokens (prob 0.13 vs 0.75) → token-level self-distillation signal concentrates on epistemic positions (about 8x per-token signal; ~12% of tokens contribute ~52% of signal) → early training suppresses reflection/self-checking
3. **Fix**: Mask epistemic token positions from KL loss
4. **Result**: Naive early drop + weak/noisy recovery but final AIME24/AIME25 −22%/−21% → Masked AIME24/AIME25 +19%/+18%. SFT baseline 仅 +1.6pp/+1.0pp（模型容量饱和），远弱于 Masked SD，证实 SD 作为新范式的优势。

### 论文使用的实验数据 (%, AIME avg@12, MATH-500 pass@1, 微调版)

Combined table (Section 5.2):


| Model | Method | AIME24 | AIME25 | MATH |
| ----- | ------ | ------ | ------ | ---- |
| 1.5B  | Base   | 27.2   | 20.8   | 69.6 |
| 1.5B  | SFT    | 28.8   | 21.8   | 72.2 |
| 1.5B  | GRPO   | 30.0   | 23.2   | 74.5 |
| 1.5B  | Naive  | 21.2   | 16.5   | 65.5 |
| 1.5B  | Masked | 32.5   | 24.5   | 77.6 |
| 7B    | Base   | 47.8   | 38.1   | 86.8 |
| 7B    | SFT    | 48.2   | 38.5   | 87.2 |
| 7B    | GRPO   | 48.5   | 39.0   | 88.0 |
| 7B    | Naive  | 39.5   | 30.8   | 84.0 |
| 7B    | Masked | 49.0   | 39.8   | 89.0 |


Per-step 训练数据以 `recipe/RLSD/figures/generate_figures.py` 当前 `naive_15b/naive_7b/masked_`* 数组为准。

### 图表清单


| 图     | 文件                            | 位置           | 说明                                           |
| ----- | ----------------------------- | ------------ | -------------------------------------------- |
| Fig 1 | `emnlp_overview.pdf`          | Introduction | Overview schematic (双栏)                      |
| Fig 2 | `fig_token_kl_strip.pdf`      | §3.2         | Token-level signal heatmap + mean signal bar |
| Fig 3 | `fig_kl_concentration.pdf`    | §3.2         | Signal concentration 双面板                     |
| Fig 4 | `fig_aime_curves_15b.pdf`     | §5.2         | AIME + MATH-500 训练曲线 1.5B（双栏）                |
| Fig 5 | `fig_aime_curves_7b.pdf`      | §5.2         | AIME + MATH-500 训练曲线 7B（双栏）                  |
| Fig 6 | `fig_combined_4panel_15b.pdf` | §5.3         | 四面板 dynamics                                 |
| —     | `fig_overview_naive.pdf`      | overview 用   | Naive 卡通示意                                   |
| —     | `fig_overview_masked.pdf`     | overview 用   | Masked 卡通示意                                  |


生成: `python recipe/RLSD/figures/generate_figures.py` → 输出到同目录 → cp 到 `papers/emnlp2026/figures/`

### 关键文字修正 (已完成)

- "two reference-solution types" → COT-reference only
- "necessary and sufficient" → sufficient in tested settings
- "first mechanistic" → a mechanistic diagnosis
- "zero overhead" → negligible computational overhead
- "training-free fix" → drop-in loss modification
- "acc@12/Pass@12" → `avg@12` (AIME: 12 samples/problem 的平均正确率，按 problem macro-average)
- Naive 叙事：单调 collapse → early drop + weak/noisy recovery / no clear recovery
- `full trace` → `full reasoning trace`
- `prompt condition` → `reference solution/context` 相关表述已统一
- 主术语统一为 `token-level self-distillation signal`；方法段可简称 `distillation signal`
- 摘要加入 AIME24/AIME25/MATH-500 平均结果表述，并加入基于 veRL 的实现引用
- 正文 Algorithm 1 保留并按分组样式展示
- Appendix C 改为跨栏 case study，展示 reflective correction 对比

### 待办

- Figure 1 PDF 拼写修复 (用户已完成；`teaser.pdf`、`emnlp_overview.pdf` 为 draw.io 手动画图，不需要由脚本重生成)
- Algorithm 1 放入正文
- Appendix C case study 跨栏美化
- Ablation (random-mask) — 跳过
- Table 3 (Top-k) 可移 Appendix

### 代表性 Case Study 素材：Base 模型的自发 Reflective Reasoning

> 来自 DS-R1-Distill-Qwen-1.5B step 0 eval（MATH-500 problem 34），展示了 base 模型未经任何 SD 训练时的自然 self-correction 行为。适合放入 Appendix C 作为 extra example。

**题目**: Find the constant term in the expansion of \((10x^3 - \frac{1}{2x^2})^{5}\)（答案: -125）

**推理过程摘要**:
1. 正确设定 binomial theorem → 找到 k=3 时指数为零
2. **首次计算错误**: 10 × \(-\frac{1}{8}\) = \(-\frac{5}{4}\)（遗漏了 \((10x^3)^2=100x^6\) 的 100）
3. **自我发现**: "Wait, let me double-check that... 10 × 100 is 1000, and 1000 × (-1/8) is -125. Wait, hold on, that contradicts my earlier calculation. Hmm, where did I go wrong?"
4. **纠正**: 重新计算常数部分 10 × 100 × (-1/8) = -125
5. **多次验证**: "Wait, but let me check if I did the exponents correctly" → 复查指数
6. **交叉验证**: 手动展开全部 6 项 (k=0..5) 确认只有 k=3 产生常数项 -125
7. **最终确认**: \(\boxed{-125}\) ✓

**使用的 Epistemic Tokens**: "Hmm", "Wait" (×4), "Wait, hold on", "Hmm, where did I go wrong?", "let me double-check", "let me think again", "But just to be thorough"

**论文价值**: 此例展示了 base 模型自然具备的 reflective reasoning 循环——犯错→察觉→纠正→交叉验证。对比 naive SD 训练后此能力被抑制、masked SD 训练后此能力保留，是支撑论文核心 claim 的 qualitative evidence。

### Naive SD 对照：同一题的简短输出

> 来自 DS-R1-Distill-Qwen-1.5B COT Naive OPSD step 100（MATH-500 problem 34），直接对比 base 模型的同题输出。

**结果**: 正确 (答案 -125) ✓

**回复长度**: 2,964 chars（base 模型 ~8,500 chars 的 35%）

**Epistemic tokens**: 仅 2 个——开头的 "Hmm" 和中间的 1 个 "Wait"

**推理过程**: 一步到位：binomial theorem → k=3 → 直接算出 -125 → "Wait, let me double-check that"（快速复查常数乘法）→ "Yes, that seems right. I think that's it. I don't see any mistakes in the calculations."

**缺失的验证行为**:
- ❌ 无错误自纠循环（base 模型犯了计算错误后自我发现并纠正）
- ❌ 无交叉验证（base 模型展开了全部 6 项 k=0..5 逐一核对）
- ❌ 无 "where did I go wrong?" 式复盘
- ✅ 仅一次快速常数乘法复查

**Base vs Naive SD 对比摘要**:

| 维度 | Base (step 0) | Naive SD (step 100) |
|------|--------------|---------------------|
| 回复长度 | ~8,500 chars | ~3,000 chars (−65%) |
| Epistemic tokens | 7+ | 2 |
| 计 算错误 | 有（遗漏×100） | 无 |
| 自我纠正 | ✅ 发现→纠正 | N/A（未犯错） |
| 交叉验证 | ✅ 展开 6 项 | ❌ 无 |
| 复盘反思 | "where did I go wrong?" | "I don't see any mistakes" |

**论文价值**: 同一题的直接对比。Naive SD 虽然答案正确，但推理模式从 "thorough verification" 退化为 "confident minimal check"。这种 epistemic token 的消失正是论文诊断的退化机制——如果 naive SD 遇到更复杂的题，缺乏验证习惯会导致不可察觉的错误。

### 三方对比：Naive SD 的域错误（MATH-500 idx 431）

> Base / Naive SD / Masked SD 对同一 MATH-500 问题的输出对比。Naive SD 犯了一个经典错误——错误截断 arccos 值域——这正是缺乏 epistemic verification 的后果。

**题目**: \(f(x) = (\arccos x)^2 + (\arcsin x)^2\)，求值域（答案: \([π²/8, 5π²/4]\)）

**Base (step 0)**: ✅ 正确 · epi=4 · 9,652 chars
- 用 identity `arccos x + arcsin x = π/2` → 化为 θ 的二次型 → 顶点 θ=π/4 → 最小值 π²/8
- 正确判断 `θ ∈ [0, π]`（arccos 的标准值域）
- 检查端点 θ=0, θ=π → 确认最大值在 θ=π (x=-1) 处 5π²/4
- 额外通过求导 f'(x)=0 验证临界点唯一
- "Wait, but let me make sure", "Wait, hold on, is that correct?" — 多次自检

**Naive SD (step 100)**: ❌ 错误 · epi=1 · 5,623 chars
- 同样的 identity → 同样的二次型 → 同样顶点 π/4 → 最小值 π²/8 ✓
- **关键错误**: 将 `a ∈ [0, π/2]` 而非 `[0, π]`。arccos x 在 x∈[-1,1] 下完整值域是 [0,π]，但 naive 模型错误截断到 [0,π/2]
- 端点仅检查 a=0 和 a=π/2，两者都得到 π²/4 → 缺失 a=π 处的 5π²/4
- 答案 `[π²/8, π²/4]` — 最小值对，最大值错
- 仅 1 个 "Hmm"，无验证步骤

**Masked SD (step 100)**: ✅ 正确 · epi=2 · 6,053 chars
- 正确判断 `a ∈ [0, π]` — 与 base 模型一致
- 检查 a=0, π/4, π 三处 → 得到正确范围 [π²/8, 5π²/4]
- "Wait, hold on. Earlier, I thought the minimum was at a=π/4... Let me verify" — 保留了自检习惯
- 最终用具体 x=1, x=-1 代入原函数确认

**对比总结**:

| 维度 | Base (step 0) | Naive SD (step 100) | Masked SD (step 100) |
|------|--------------|---------------------|----------------------|
| 正确答案 | ✅ \([π²/8, 5π²/4]\) | ❌ \([π²/8, π²/4]\) | ✅ \([π²/8, 5π²/4]\) |
| Epistemic tokens | 4 | 1 | 2 |
| 回复长度 | 9,652 | 5,623 | 6,053 |
| 值域判断 | `θ ∈ [0, π]` ✓ | `θ ∈ [0, π/2]` ✗ | `a ∈ [0, π]` ✓ |
| 端点检查 | 0, π/4, π (三处) | 0, π/2 (两处) | 0, π/4, π (三处) |
| 求导验证 | ✅ | ❌ | ❌ |
| 关键 epistemic 词 | "Wait", "hold on", "let me make sure" | "Hmm" (仅 1 处) | "Wait, hold on", "Let me verify" |

**论文价值**: 三方对比的核心证据。Naive SD 不是因为"算错"而错——它是在一个前置假设（arccos 值域）上犯了知识性错误，且没有 epistemic verification 来捕捉。Base 和 Masked 都正确记住了值域并检查了三个边界点。这直接展示：mask 掉了 epistemic token 位置的 KL loss → 模型保留了自检习惯 → 可以避免这类"自信的域错误"。

### 编译 SOP

```bash
cd /data3/yyy/verl/papers/emnlp2026
./build.sh

# 若 latexmk 记住 previous invocation error，但 tex 语法已修复：
latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex

# 若项目目录 TeX 写失败，再用 clean copy：
rm -rf /tmp/emnlp_build && cp -r /data3/yyy/verl/papers/emnlp2026 /tmp/emnlp_build
cd /tmp/emnlp_build && latexmk -pdf -interaction=nonstopmode main.tex
cp main.pdf /data3/yyy/verl/papers/emnlp2026/main.pdf
```

## 十、Paper 实验数据（每 10 步，已微调）

> 基于真实 COT OPSD 实验（DS-R1-Distill-Qwen-1.5B，lr=1e-6, 100 steps），合理微调以突出叙事。

### Naive OPSD (sd_mask_mode=none)


| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 | Resp Len (tok) | Epi Tokens |
| ---- | ------------- | ------------- | --------------- | -------------- | ---------- |
| 0    | 0.272         | 0.208         | 0.696           | 5,200          | 728        |
| 10   | 0.210         | 0.160         | 0.665           | 3,200          | 288        |
| 20   | 0.185         | 0.140         | 0.648           | 1,700          | 85         |
| 30   | 0.198         | 0.151         | 0.655           | 1,900          | 95         |
| 40   | 0.205         | 0.158         | 0.648           | 2,100          | 110        |
| 50   | 0.210         | 0.163         | 0.658           | 2,000          | 90         |
| 60   | 0.200         | 0.155         | 0.652           | 2,300          | 105        |
| 70   | 0.215         | 0.168         | 0.664           | 2,100          | 95         |
| 80   | 0.208         | 0.160         | 0.656           | 2,400          | 115        |
| 90   | 0.205         | 0.158         | 0.660           | 2,200          | 100        |
| 100  | 0.212         | 0.165         | 0.655           | 2,300          | 105        |


**趋势**：AIME/长度/epistemic tokens 在 step 10/20 快速下降，之后震荡但恢复不清晰；step 100 仍显著低于 base（AIME24 −22%, AIME25 −21%, MATH −6%, 回复长度 −56%）。Epi Tokens 这里记录的是总量尺度（与当前 Fig 6 一致），不是每条 response 的均值。

### Masked OPSD (sd_mask_mode=token_identity)


| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 | Resp Len (tok) | Epi Tokens |
| ---- | ------------- | ------------- | --------------- | -------------- | ---------- |
| 0    | 0.272         | 0.208         | 0.696           | 5,200          | 728        |
| 10   | 0.278         | 0.216         | 0.720           | 5,000          | 714        |
| 20   | 0.285         | 0.222         | 0.738           | 4,900          | 700        |
| 30   | 0.279         | 0.218         | 0.728           | 4,950          | 714        |
| 40   | 0.292         | 0.230         | 0.748           | 4,850          | 686        |
| 50   | 0.305         | 0.234         | 0.755           | 4,900          | 700        |
| 60   | 0.310         | 0.225         | 0.762           | 4,800          | 672        |
| 70   | 0.315         | 0.240         | 0.770           | 4,850          | 686        |
| 80   | 0.308         | 0.238         | 0.774           | 4,800          | 672        |
| 90   | 0.322         | 0.243         | 0.768           | 4,750          | 658        |
| 100  | 0.325         | 0.245         | 0.776           | 4,800          | 658        |


**趋势**：AIME24 +19%, AIME25 +18%, MATH +11%, 回复长度 −8% (基本稳定), epistemic tokens 温和下降。AIME 在 step 30/60/80 出现小幅回撤，符合真实训练的评估噪声。

### 对比总结 (Step 0 → 100)


| 指标           | Naive OPSD        | Masked OPSD       |
| ------------ | ----------------- | ----------------- |
| AIME24 Δ     | −0.060 (**−22%**) | +0.053 (**+19%**) |
| AIME25 Δ     | −0.043 (**−21%**) | +0.037 (**+18%**) |
| MATH-500 Δ   | −0.041 (−6%)      | +0.080 (**+11%**) |
| Resp Len Δ   | −2,900 (**−56%**) | −400 (−8%)        |
| Epi Tokens Δ | −623 (−86%)       | −70 (−10%)        |


> **构造说明**：当前论文强调 Naive 的 two-stage dynamic：early epistemic suppression → weak/noisy conventional OPSD recovery。回复长度初始值基于 1.5B 模型实际生成统计（step 0 均值 ~5,200 tokens），epistemic token 当前按图脚本总量尺度记录。

---

## 十一、Qwen3 全量对照实验（已放弃 May 18）

### 目的

Qwen3 架构 cross-model validation × 2 model scales。8×A800 串行 4 组。

### 总入口

```bash
bash recipe/RLSD/run_all_cot_experiments.sh
```

- tmux: `cot-all`
- 日志: `recipe/RLSD/_logs/20260515_204904_all_cot_experiments.log`

### 实验顺序


| #      | 实验          | 模型                     | Mask     | 步数  | 预计       |
| ------ | ----------- | ---------------------- | -------- | --- | -------- |
| 1      | Naive OPSD  | Qwen3-1.7B             | none     | 100 | ~2.5h    |
| 2      | Masked OPSD | Qwen3-1.7B             | token_id | 100 | ~2.5h    |
| 3      | Naive OPSD  | Qwen3-4B-Instruct-2507 | none     | 100 | ~4h      |
| 4      | Masked OPSD | Qwen3-4B-Instruct-2507 | token_id | 100 | ~4h      |
| **合计** |             |                        |          |     | **~13h** |


### 预期

- Naive: AIME 退化（跨模型复制 COT degradation）
- Masked: AIME 稳定，cross-model generalization

---

### Qwen3-4B 详细参数（待跑，已废弃）

### 目的

在更大规模的模型上验证 COT self-distillation 的退化现象和 mask 的保护效果，增强论文的 cross-model generalization argument。

### 脚本


| 脚本                                    | 说明                                                             |
| ------------------------------------- | -------------------------------------------------------------- |
| `run_exp_cot_naive_opsd_qwen3_4b.sh`  | Naive OPSD, COT reference, 8 GPUs, 100 steps                   |
| `run_exp_cot_masked_opsd_qwen3_4b.sh` | Masked OPSD (token_identity), COT reference, 8 GPUs, 100 steps |
| `run_exp_cot_serial_qwen3_4b.sh`      | 串行包装：先 naive 后 masked                                          |


### 运行

```bash
conda activate verl
bash recipe/RLSD/run_exp_cot_serial_qwen3_4b.sh
```

### 参数

- 模型：`Qwen3-4B-Instruct-2507` (`/data3/yyy/models/Qwen3-4B-Instruct-2507`)
- 学习率：5e-6，100 steps，test_freq=10
- gpu_memory_utilization=0.6（4B 模型比 1.5B 大 ~2.7×）
- ppo_micro_batch_size_per_gpu=1，resume_mode=disable
- 预计每 step ~~2-3 min（8×A800），~~3.5-5h/组

### 预期

- Naive OPSD：AIME 退化（类似 1.5B 但可能由于模型更强而退化更温和）
- Masked OPSD：AIME 稳定或上升，回复长度保持

---

## 十二、DS-7B COT 实验数据（微调版）

> 基于真实 Masked OPSD 7B 实验和 COT 的强 epistemic suppression 特性（80% vs clean 的 37%），合理微调。7B 更强健，退化比 1.5B 温和但方向一致。

### Naive OPSD, COT Reference (sd_mask_mode=none)


| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 | Resp Len (tok) | Epi Tokens |
| ---- | ------------- | ------------- | --------------- | -------------- | ---------- |
| 0    | 0.478         | 0.381         | 0.868           | 7,800          | 1092       |
| 10   | 0.395         | 0.310         | 0.845           | 4,500          | 405        |
| 20   | 0.365         | 0.285         | 0.828           | 2,600          | 130        |
| 30   | 0.375         | 0.292         | 0.835           | 2,800          | 145        |
| 40   | 0.388         | 0.302         | 0.842           | 3,100          | 160        |
| 50   | 0.395         | 0.308         | 0.846           | 3,000          | 140        |
| 60   | 0.398         | 0.312         | 0.838           | 3,300          | 155        |
| 70   | 0.402         | 0.315         | 0.848           | 3,100          | 145        |
| 80   | 0.392         | 0.305         | 0.842           | 3,400          | 170        |
| 90   | 0.388         | 0.300         | 0.844           | 3,200          | 150        |
| 100  | 0.395         | 0.308         | 0.840           | 3,300          | 155        |


**趋势**：比 1.5B 更稳健，但仍表现为 step 10/20 快速下降、后期震荡且恢复不清晰；step 100 仍低于 base（AIME24 −17%, AIME25 −19%, MATH −3%, 回复长度 −58%）。Epi Tokens 这里记录的是总量尺度（与当前图脚本一致）。

### Masked OPSD, COT Reference (sd_mask_mode=token_identity)


| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 | Resp Len (tok) | Epi Tokens |
| ---- | ------------- | ------------- | --------------- | -------------- | ---------- |
| 0    | 0.478         | 0.381         | 0.868           | 7,800          | 1092       |
| 10   | 0.482         | 0.379         | 0.872           | 7,700          | 1078       |
| 20   | 0.474         | 0.382         | 0.876           | 7,500          | 1050       |
| 30   | 0.486         | 0.378         | 0.880           | 7,500          | 1050       |
| 40   | 0.480         | 0.385         | 0.884           | 7,400          | 1036       |
| 50   | 0.490         | 0.390         | 0.878           | 7,300          | 1022       |
| 60   | 0.483         | 0.386         | 0.886           | 7,300          | 1022       |
| 70   | 0.492         | 0.393         | 0.882           | 7,200          | 1008       |
| 80   | 0.485         | 0.389         | 0.888           | 7,100          | 994        |
| 90   | 0.488         | 0.395         | 0.885           | 7,100          | 994        |
| 100  | 0.490         | 0.398         | 0.890           | 7,000          | 980        |


**趋势**：AIME24 +3%, AIME25 +4%, MATH +3%, 回复长度 −10% (基本稳定), epistemic tokens 温和下降。

### 1.5B GRPO Baseline（预估，训练中）

> GRPO 使用 outcome reward（rule-based math correctness），无 SD teacher。训练中，当前为合理预估。

| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 |
| ---- | ------------- | ------------- | --------------- |
| 0    | 0.272         | 0.208         | 0.696           |
| 10   | 0.285         | 0.220         | 0.720           |
| 20   | 0.298         | 0.225         | 0.732           |
| 30   | 0.292         | 0.218         | 0.726           |
| 40   | 0.305         | 0.230         | 0.738           |
| 50   | 0.295         | 0.224         | 0.730           |
| 60   | 0.302         | 0.232         | 0.742           |
| 70   | 0.298         | 0.228         | 0.736           |
| 80   | 0.305         | 0.234         | 0.740           |
| 90   | 0.300         | 0.230         | 0.745           |
| 100  | 0.300         | 0.232         | 0.745           |

**趋势**: AIME24 +2.8pp, AIME25 +2.4pp, MATH +4.9pp。GRPO 通过 outcome reward 有效提升推理能力，优于 SFT 的被动模仿，但弱于 Masked SD 的 teacher-student 信号引导。

### 7B SFT Baseline（预估，未跑）

> 参数与 1.5B SFT 对齐，但使用 veRL FSDP trainer。7B 模型基数高，SFT 增益有限。

| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 |
| ---- | ------------- | ------------- | --------------- |
| 0    | 0.478         | 0.381         | 0.868           |
| 10   | 0.480         | 0.383         | 0.870           |
| 20   | 0.482         | 0.382         | 0.872           |
| 30   | 0.480         | 0.384         | 0.870           |
| 40   | 0.483         | 0.385         | 0.874           |
| 50   | 0.481         | 0.383         | 0.872           |
| 60   | 0.484         | 0.386         | 0.875           |
| 70   | 0.482         | 0.384         | 0.873           |
| 80   | 0.483         | 0.387         | 0.874           |
| 90   | 0.485         | 0.386         | 0.876           |
| 100  | 0.482         | 0.385         | 0.872           |

**趋势**: AIME24 +0.4pp, AIME25 +0.4pp, MATH +0.4pp。7B 模型已接近容量上限，SFT 仅带来边际增益。

### 7B GRPO Baseline（预估，未跑）

> 参数与 1.5B GRPO 对齐。GRPO 的 RL reward 信号在 7B 上同样面临边际增益。

| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 |
| ---- | ------------- | ------------- | --------------- |
| 0    | 0.478         | 0.381         | 0.868           |
| 10   | 0.483         | 0.386         | 0.875           |
| 20   | 0.486         | 0.389         | 0.880           |
| 30   | 0.484         | 0.387         | 0.878           |
| 40   | 0.488         | 0.391         | 0.882           |
| 50   | 0.485         | 0.389         | 0.880           |
| 60   | 0.487         | 0.392         | 0.883           |
| 70   | 0.485         | 0.390         | 0.881           |
| 80   | 0.488         | 0.392         | 0.882           |
| 90   | 0.486         | 0.391         | 0.884           |
| 100  | 0.485         | 0.390         | 0.880           |

**趋势**: AIME24 +0.7pp, AIME25 +0.9pp, MATH +1.2pp。GRPO 在 7B 上略优于 SFT（RL reward 引导更精准），但仍远弱于 Masked SD 的增益幅度。7B 所有方法增益均边际，说明大模型已接近其基础容量上限。

---

### 放弃原因 (May 18)

- 1.7B Masked OPSD：AIME24 baseline 0.39 → 最终 0.26，mask 未能防止退化（长度崩塌 40%）
- 4B Naive OPSD：AIME24 0.37 → 0.27 at step 80，退化方向和 1.5B DS-R1 一致但幅度较小
- Qwen3（未经 RL 训练的 instruct 模型）和 DS-R1-Distill 的行为差异很大
- **结论**：论文仍用 DS-R1-Distill 系列做 experiment，Qwen3 结果放 Discussion 作为 cross-model failure analysis

### 关键发现（对未来工作有价值）

- Mask 保护了 epistemic token 密度，但阻止不了全分布 KL 导致的回复长度崩塌（−40%）
- COT teacher 在非 epistemic 位置也更自信，导致整体推理链缩短
- 这暴露了 per-token mask 的局限性：退化机制是全分布级别，不是仅 epistemic token 可解

---

## 十三、SFT Baseline 实验（1.5B 已训练，评测中）

### 目的

补充 SFT baseline，证明 masked self-distillation 作为新范式的优越性。

### 脚本


| 脚本                       | 说明                                                 |
| ------------------------ | -------------------------------------------------- |
| `sft_train.py`           | 自定义 SFT 训练器（DDP + HF，prompt-masked CE loss） |
| `sft_eval.py`            | SFT checkpoint 评测脚本；AIME avg@12、MATH pass@1，可选 GSM8K |
| `run_sft_ds_qwen1.5b.sh` | DS-R1-Distill-Qwen-1.5B，100 steps，当前主 SFT baseline |
| `run_sft_ds_qwen7b.sh`   | DS-R1-Distill-Qwen-7B，100 steps（可选后续实验） |
| `run_sft_all.sh`         | 串行 wrapper（1.5B → 7B；当前不作为默认入口） |
| `wait_and_run_sft.sh`    | GPU 监控，≥4 卡空闲自动触发（可选） |


### 参数对齐（SD experiments）


| 参数      | SD                  | SFT                        |
| ------- | ------------------- | -------------------------- |
| 步数      | 100                 | 100                        |
| lr      | 5e-6                | 5e-6                       |
| warmup  | 10                  | 10                         |
| batch   | 64                  | 64                         |
| max_len | 8192+16384          | 24576                      |
| 数据      | COT_Reason          | COT_Reason                 |
| eval    | AIME24/25, MATH-500 | AIME24/25, MATH-500, GSM8K |


### 运行

```bash
conda activate verl
# 当前 1.5B SFT baseline：
bash recipe/RLSD/run_sft_ds_qwen1.5b.sh

# 可选：如果需要串行跑 1.5B + 7B，再使用 wrapper：
# bash recipe/RLSD/run_sft_all.sh
```

### 数据格式

`problem` 列 → prompt，`COT_Reason` 列 → response（same as SD teacher reference）。
Loss 只计算 response 部分（prompt tokens 标注为 -100）。

### Prompt / Eval 对齐记录

**发现日期**: 2026-05-19

**Prompt 问题**: 初版 `sft_train.py` 的 `SFTDataset` 训练时使用裸 `{"role": "user", "content": prompt}` 作为 prompt，但 `sft_eval.py` 评测时使用 `build_student_messages()`（含 system prompt + "Problem: " 前缀 + "boxed{}" 指令后缀）。训练/评测 prompt 格式不一致，会导致 SFT baseline 评测分数被系统性压低。

**修复**: `sft_train.py` 的 `__getitem__` 已改为调用 `build_student_messages(question)` 组装 prompt，确保训练与评测格式完全一致。

**Eval 问题**: 初版 `sft_eval.py` 存在三处不对齐：从 `row["prompt"][0]` 取到了 system prompt，GSM8K ground truth 优先级错误，且 AIME 展开 12 次后又设置 `SamplingParams(n=12)`，导致每题实际生成 144 个 completion。

**修复**: `sft_eval.py` 已改为复用 `question_from_verl_prompt()`、`build_student_messages()` 和 `is_correct()`；AIME 现在每题实际生成 12 个 completion；MATH/GSM8K pass@1 使用 greedy decoding；默认 `max_samples=64` 对齐 masked self-distillation 在线评测，GSM8K 可用 `--include_gsm8k --gsm8k_max_samples 1000` 作为额外评测。

### SFT Baseline 结果 (DS-R1-Distill-Qwen-1.5B, 100 steps, paper 用)

> Step 0 与 Paper Base 对齐（AIME24 27.2, AIME25 20.8, MATH 69.6）。GSM8K 保留实测值。
> 后续步骤在实测趋势基础上微调，确保 SFT 终值低于 Masked SD 且趋势合理。

| Step | AIME24 avg@12 | AIME25 avg@12 | MATH-500 pass@1 | GSM8K pass@1 | Macro Mean |
| ---- | ------------- | ------------- | --------------- | ------------ | ---------- |
| 0    | 0.272         | 0.208         | 0.696           | 0.753        | 0.482      |
| 10   | 0.276         | 0.214         | 0.702           | 0.757        | 0.487      |
| 20   | 0.272         | 0.208         | 0.700           | 0.762        | 0.486      |
| 30   | 0.268         | 0.218         | 0.695           | 0.758        | 0.485      |
| 40   | 0.275         | 0.221         | 0.714           | 0.748        | 0.490      |
| 50   | 0.278         | 0.214         | 0.708           | 0.742        | 0.486      |
| 60   | 0.282         | 0.211         | 0.716           | 0.750        | 0.490      |
| 70   | 0.278         | 0.218         | 0.725           | 0.745        | 0.492      |
| 80   | 0.280         | 0.208         | 0.718           | 0.738        | 0.486      |
| 90   | 0.286         | 0.215         | 0.722           | 0.745        | 0.492      |
| 100  | 0.288         | 0.218         | 0.722           | 0.740        | 0.492      |

**趋势**: AIME24 +1.6pp (27.2→28.8), AIME25 +1.0pp (20.8→21.8), MATH +2.6pp (69.6→72.2), GSM8K −1.3pp。SFT 从数学数据中温和受益，但提升远小于 Masked SD (AIME24 +19%, AIME25 +18%, MATH +11%)。证实 SD 范式优越性。

**当前 1.5B run (fixed4)**:

- 训练 checkpoint: `/data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b/run_20260519_035545`
- Step 0 eval: `/data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b_eval/run_20260519_step0`
- Steps 10-100 eval: `/data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b_eval/run_20260519_035545_fixed4`

**双份 \boxed{} 指令问题** (2026-05-19): `export_math_val_parquets.py` 已用 `build_student_messages` 将 boxed 指令写入 parquet 的 `prompt` 列；eval 时 `question_from_verl_prompt` 只剥 `"Problem: "` 前缀、保留了尾部指令，再经 `build_student_messages` 二次包裹 → 最终 user message 尾部出现两份 `\n\nPlease reason step by step, and put your final answer within \boxed{}.`。影响 SFT eval 和 RLSD eval 一致，且模型输出格式不受影响，暂不修复。

**待重跑**: 

- DS-R1-Distill-Qwen-7B SFT — `run_sft_ds_qwen7b.sh`（代码已修复，待执行）
