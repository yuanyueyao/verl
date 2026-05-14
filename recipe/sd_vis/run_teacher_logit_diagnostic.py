"""
诊断实验：Teacher 在 student 错误轨迹上的 per-token logits 质量。

1. 从 parquet 取题，student prompt 生成错误回复
2. 对比 Student / Teacher 在错误回复上的逐 token 预测熵
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recipe.RLSD.rlsd.prompt import (
    build_student_messages,
    build_teacher_privileged_messages,
)

MODEL_PATH = "/data3/yyy/models/Qwen2.5-3B-Instruct"
PARQUET = "/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet"
N_SAMPLES = 3
MAX_NEW = 512
ANALYZE_TOKENS = 200  # 分析错误回复前 200 token

# ── 加载数据 & 模型 ────────────────────────────────────────
print("加载数据...", flush=True)
df = pd.read_parquet(PARQUET)

print("加载模型...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
model.eval()
print("OK\n")


def generate_response(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW,
                             do_sample=True, temperature=0.7, top_p=0.9)
    resp_ids = out[0, n_in:]
    return tokenizer.decode(resp_ids, skip_special_tokens=True), resp_ids.tolist()


@torch.no_grad()
def per_token_entropy(context_text, response_ids):
    """给定 context + response tokens，返回每个 response token 位置的预测熵。"""
    # 将 context 文本 + response 文本拼接做 forward
    full_text = context_text + tokenizer.decode(response_ids, skip_special_tokens=True)
    enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                    max_length=32768).to(model.device)
    ctx_len = len(tokenizer(context_text, return_tensors="pt")["input_ids"][0])
    # 确保没有因截断而丢失数据
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits[0]  # (seq, V)
    # 预测 token[t] 用 logits[t-1]
    resp_logits = logits[ctx_len - 1: ctx_len - 1 + len(response_ids), :]
    probs = torch.softmax(resp_logits.float(), dim=-1)
    log_probs = torch.log_softmax(resp_logits.float(), dim=-1)
    ent = -(probs * log_probs).sum(dim=-1).cpu().numpy()  # (T,)
    return ent


for i in range(N_SAMPLES):
    row = df.iloc[i * 500]  # 取不同位置，增加多样性
    question = str(row["problem"])
    gt = str(row["Answer"])
    solution = str(row.get("solution", ""))

    print(f"=== 样本 {i} ===")
    print(f"  question[:100]: {question[:100]}...")
    print(f"  gt: {gt}")
    print(f"  solution: {len(solution)} chars")

    # Step 1: 用 Student prompt 生成一条（可能错误的）回复
    s_msgs = build_student_messages(question)
    resp_text, resp_ids = generate_response(s_msgs)
    resp_ids = resp_ids[:ANALYZE_TOKENS]
    print(f"  response: {len(resp_ids)} tokens")

    # Step 2: Student context 下计算逐 token 熵
    s_text = tokenizer.apply_chat_template(s_msgs, tokenize=False,
                                            add_generation_prompt=True)
    s_ent = per_token_entropy(s_text, resp_ids)

    # Step 3: Teacher context（含 solution）下计算逐 token 熵
    t_msgs = build_teacher_privileged_messages(question, gt, solution)
    t_text = tokenizer.apply_chat_template(t_msgs, tokenize=False,
                                            add_generation_prompt=True)
    ctx_tok = len(tokenizer(t_text, return_tensors="pt")["input_ids"][0])
    print(f"  Teacher context: {ctx_tok} tokens (含 {len(solution)} chars solution)")

    t_ent = per_token_entropy(t_text, resp_ids)

    assert len(s_ent) == len(t_ent), f"token mismatch: {len(s_ent)} vs {len(t_ent)}"

    # Step 4: 对比
    diff = t_ent - s_ent
    print(f"  Student entropy: mean={s_ent.mean():.3f}  max={s_ent.max():.2f}")
    print(f"  Teacher entropy: mean={t_ent.mean():.3f}  max={t_ent.max():.2f}")
    print(f"  Δ(T-S): mean={diff.mean():+.3f}  max={diff.max():+.2f}  "
          f">0 token 占比: {100*(diff>0).mean():.0f}%")
    print(f"  Teacher 高熵 token(>5): {(t_ent > 5).sum()}/{len(t_ent)}")
    print(f"  Student 高熵 token(>5): {(s_ent > 5).sum()}/{len(s_ent)}")

    # 找 Teacher 熵明显更高的位置
    spike = (diff > 1.5).nonzero()[0]
    if len(spike) > 0:
        print(f"  Teacher 熵尖峰(Δ>1.5) 位置: {spike[:8].tolist()}")
        for idx in spike[:3]:
            ctx = resp_ids[max(0,idx-3):idx+3]
            ctx_text = tokenizer.decode(ctx)
            print(f"    [{idx}] s={s_ent[idx]:.2f} t={t_ent[idx]:.2f}  ctx: {repr(ctx_text)[:80]}")
    else:
        print(f"  无显著 Teacher 熵尖峰 (Δ>1.5)")

    # 检查 Teacher 熵是否在某些位置特别低（过度自信）
    dip = (diff < -1.5).nonzero()[0]
    if len(dip) > 0:
        print(f"  Teacher 低熵谷(Δ<-1.5) 位置: {dip[:5].tolist()}")
    print()

print("分析完成")
