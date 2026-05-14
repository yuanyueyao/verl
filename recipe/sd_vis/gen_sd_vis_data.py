"""
生成 SD 可视化数据：对每个 response token，计算 teacher 和 student 的 top-k 概率分布。
输出 JSON 供 HTML 交互展示。
"""

import json, sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from recipe.RLSD.rlsd.prompt import (
    build_student_messages,
    build_teacher_privileged_messages,
)

MODEL_PATH = "/data3/yyy/models/Qwen2.5-3B-Instruct"
PARQUET = "/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet"
OUTPUT = Path(__file__).parent / "sd_vis_data.json"

TOP_K = 20
MAX_RESP_TOKENS = 8192  # 完整回复


def main():
    # ── 加载 ──
    df = pd.read_parquet(PARQUET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    vocab = tokenizer.get_vocab()
    id2tok = {v: k for k, v in vocab.items()}

    def clean_token(tok_str: str) -> str:
        """将 tokenizer 的原始 token 转为可读显示。"""
        return tok_str.replace("Ġ", " ").replace("Ċ", "\\n")  # Ġ→空格

    # ── 选一道有完整 solution 的题 ──
    for i in range(len(df)):
        sol = str(df.iloc[i].get("solution", ""))
        if len(sol) > 500:
            row = df.iloc[i]
            break

    question = str(row["problem"])
    gt = str(row["Answer"])
    solution = str(row.get("solution", ""))
    print(f"题目: {question[:100]}...")
    print(f"答案: {gt}")
    print(f"solution: {len(solution)} chars")

    # ── Student 生成回复（greedy）──
    s_msgs = build_student_messages(question)
    s_text = tokenizer.apply_chat_template(s_msgs, tokenize=False,
                                            add_generation_prompt=True)
    s_inputs = tokenizer(s_text, return_tensors="pt").to(model.device)
    s_prompt_len = s_inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**s_inputs, max_new_tokens=MAX_RESP_TOKENS,
                             do_sample=False, temperature=1.0,
                             pad_token_id=tokenizer.eos_token_id)
    resp_ids = out[0, s_prompt_len:].tolist()
    resp_tokens = [clean_token(id2tok.get(tid, f"<{tid}>")) for tid in resp_ids]
    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
    print(f"生成 {len(resp_ids)} tokens")

    # ── Student forward ──
    s_full_ids = out[0].unsqueeze(0)
    s_full_mask = torch.ones_like(s_full_ids)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        s_output = model(input_ids=s_full_ids, attention_mask=s_full_mask)
    s_logits = s_output.logits[0].float()  # (seq, V)

    # ── Teacher 自己生成回复（solution-conditioned，greedy）──
    t_msgs = build_teacher_privileged_messages(question, gt, solution)
    t_text = tokenizer.apply_chat_template(t_msgs, tokenize=False,
                                            add_generation_prompt=True)
    t_inputs = tokenizer(t_text, return_tensors="pt").to(model.device)
    t_prompt_len = t_inputs["input_ids"].shape[1]

    with torch.no_grad():
        t_out = model.generate(**t_inputs, max_new_tokens=MAX_RESP_TOKENS,
                               do_sample=False, temperature=1.0,
                               pad_token_id=tokenizer.eos_token_id)
    t_resp_ids = t_out[0, t_prompt_len:].tolist()
    t_resp_tokens = [clean_token(id2tok.get(tid, f"<{tid}>")) for tid in t_resp_ids]
    t_resp_text = tokenizer.decode(t_resp_ids, skip_special_tokens=True)
    print(f"Teacher 生成 {len(t_resp_ids)} tokens")

    # ── Teacher 在 Student 错误回复上的 forward ──
    t_full_text = t_text + resp_text
    t_enc = tokenizer(t_full_text, return_tensors="pt", truncation=True,
                      max_length=32768).to(model.device)
    t_prompt_len_for_logits = len(tokenizer(t_text, return_tensors="pt")["input_ids"][0])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        t_output = model(input_ids=t_enc["input_ids"],
                         attention_mask=t_enc["attention_mask"])
    t_logits = t_output.logits[0].float()  # (seq, V)

    # ── 逐 token 提取 top-k ──
    tokens_data = []
    for pos in range(len(resp_ids)):
        # Student: 预测 token[pos] 用的是 logits[s_prompt_len + pos - 1]
        s_idx = s_prompt_len + pos - 1
        s_l = s_logits[s_idx]  # (V,)
        s_probs = F.softmax(s_l, dim=-1)
        s_top_vals, s_top_ids = torch.topk(s_probs, TOP_K)
        s_top_probs = [{"token": clean_token(id2tok.get(tid.item(), f"<{tid.item()}>")),
                         "prob": round(v.item(), 6),
                         "logit": round(s_l[tid].item(), 2)}
                        for tid, v in zip(s_top_ids, s_top_vals)]

        # Teacher: 预测 token[pos] 用的是 logits[t_prompt_len_for_logits + pos - 1]
        t_idx = t_prompt_len_for_logits + pos - 1
        t_l = t_logits[t_idx]
        t_probs = F.softmax(t_l, dim=-1)
        t_top_vals, t_top_ids = torch.topk(t_probs, TOP_K)
        t_top_probs = [{"token": clean_token(id2tok.get(tid.item(), f"<{tid.item()}>")),
                         "prob": round(v.item(), 6),
                         "logit": round(t_l[tid].item(), 2)}
                        for tid, v in zip(t_top_ids, t_top_vals)]

        # 熵
        s_ent = -(s_probs * torch.log(s_probs + 1e-12)).sum().item()
        t_ent = -(t_probs * torch.log(t_probs + 1e-12)).sum().item()

        # per-token full-distribution KL(Student || Teacher)
        # D_KL(S||T) = Σ_v S(v) * log(S(v)/T(v))
        t_probs_clamped = t_probs.clamp(min=1e-12)
        s_probs_clamped = s_probs.clamp(min=1e-12)
        token_kl = (s_probs_clamped * (torch.log(s_probs_clamped) - torch.log(t_probs_clamped))).sum().item()

        # 实际 token 的概率
        actual_tok = resp_ids[pos]
        s_actual_prob = s_probs[actual_tok].item()
        t_actual_prob = t_probs[actual_tok].item()

        tokens_data.append({
            "pos": pos,
            "token": resp_tokens[pos],
            "token_id": resp_ids[pos],
            "s_entropy": round(s_ent, 4),
            "t_entropy": round(t_ent, 4),
            "kl_divergence": round(token_kl, 4),
            "s_actual_prob": round(s_actual_prob, 6),
            "t_actual_prob": round(t_actual_prob, 6),
            "s_topk": s_top_probs,
            "t_topk": t_top_probs,
        })

    # ── 保存 ──
    data = {
        "question": question,
        "ground_truth": gt,
        "solution": solution,
        "student_response_text": resp_text,
        "student_response_tokens": resp_tokens,
        "teacher_response_text": t_resp_text,
        "teacher_response_tokens": t_resp_tokens,
        "tokens_data": tokens_data,
        "top_k": TOP_K,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存到 {OUTPUT}  ({len(tokens_data)} tokens)")


if __name__ == "__main__":
    main()
