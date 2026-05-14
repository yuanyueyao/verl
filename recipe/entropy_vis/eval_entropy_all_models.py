"""
多模型逐 token 熵评测（vLLM 加速版）。

对每个模型 greedy 生成，记录每个生成位置的 top-K logprobs 以及由 top-K 近似的熵。
熵近似公式：把 top-K 之外的剩余概率质量均匀分布到剩余词表上，公式：
    H ≈ -Σ_{i∈topK} p_i·log(p_i)  +  r·(log(V-K) - log(r))，  其中 r = 1 - Σ p_i
当 K 较大（默认 50）时此近似与真实熵高度接近，且远快于 HF 全 forward。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = "/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet"
DEFAULT_OUTPUT = Path(__file__).parent / "entropy_all_models.json"

ALL_MODELS = [
    ("Qwen2.5-3B-Instruct", "/data3/yyy/models/Qwen2.5-3B-Instruct"),
    ("Qwen2.5-7B", "/data3/yyy/models/Qwen2.5-7B"),
    ("Qwen3-4B-Base", "/data3/yyy/models/Qwen3-4B-Base"),
    ("Qwen3-4B-Instruct-2507", "/data3/yyy/models/Qwen3-4B-Instruct-2507"),
    ("Qwen3-8B-Base", "/data3/yyy/models/Qwen3-8B-Base"),
    ("DeepSeek-R1-Distill-Qwen-1.5B", "/data3/yyy/models/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("DeepSeek-R1-Distill-Qwen-7B", "/data3/yyy/models/DeepSeek-R1-Distill-Qwen-7B"),
]


# ─────────────────────────── helpers ───────────────────────────


def extract_boxed(text: str) -> str | None:
    """提取最后一个 \\boxed{...}（支持嵌套大括号）。"""
    key = r"\boxed{"
    last = text.rfind(key)
    if last < 0:
        return None
    i = last + len(key)
    depth = 1
    buf = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            buf.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            buf.append(c)
        else:
            buf.append(c)
        i += 1
    return "".join(buf).strip() if depth == 0 else None


def approx_entropy_from_topk(logprobs: list[float], vocab_size: int) -> float:
    """top-K 近似熵：H ≈ -Σ p log p + r·(log(V-K) - log r)，r 为剩余质量。"""
    if not logprobs:
        return 0.0
    # logprobs 来自模型，自然概率和不超过 1（可能因数值原因略大于 1）
    sum_p = 0.0
    H = 0.0
    for lp in logprobs:
        p = math.exp(lp)
        sum_p += p
        H -= p * lp
    rest = max(0.0, 1.0 - sum_p)
    V_rest = max(1, vocab_size - len(logprobs))
    if rest > 1e-12 and V_rest > 0:
        H += rest * (math.log(V_rest) - math.log(rest))
    return H


def decode_token(tokenizer, tid: int) -> str:
    """安全 decode 单个 token，保留可见空白/换行。"""
    try:
        s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception:
        s = f"<{tid}>"
    return s


@dataclass
class Problem:
    idx: int
    question: str
    ground_truth: str


def load_problems(parquet_path: str, n: int, stride: int = 500) -> list[Problem]:
    df = pd.read_parquet(parquet_path)
    out = []
    for i in range(n):
        row = df.iloc[i * stride]
        out.append(Problem(idx=i, question=str(row["problem"]), ground_truth=str(row["Answer"])))
    return out


# ─────────────────────────── eval one model ───────────────────────────


def eval_model(
    model_name: str,
    model_path: str,
    problems: list[Problem],
    max_new: int,
    top_k: int,
    gpu_mem_util: float,
):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompts = []
    for p in problems:
        msgs = [{
            "role": "user",
            "content": (
                f"Problem: {p.question}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        }]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    # vLLM 限制：logprobs ≤ 20，否则触发 ValueError；我们用 20 作为上限
    eff_top_k = min(top_k, 20)
    if eff_top_k < top_k:
        print(f"  [warn] vLLM 限制 logprobs ≤ 20，实际使用 K={eff_top_k}（你请求 {top_k}）")

    max_model_len = max_new + 1024
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
        disable_log_stats=True,
        enforce_eager=False,
    )
    vocab_size = llm.llm_engine.model_config.get_vocab_size()

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new,
        logprobs=eff_top_k,
    )

    outputs = llm.generate(prompts, sp, use_tqdm=False)

    problem_records = []
    for prob, out in zip(problems, outputs):
        comp = out.outputs[0]
        resp_ids = list(comp.token_ids)
        resp_text = comp.text
        ans = extract_boxed(resp_text)
        correct = ans == prob.ground_truth if ans else False

        tokens_data = []
        cum_char = 0  # 累计字符 offset，用于跨模型字符级对齐
        for pos, lp_dict in enumerate(comp.logprobs or []):
            if pos >= len(resp_ids):
                break
            actual_tid = resp_ids[pos]
            # lp_dict: token_id -> Logprob(logprob, decoded_token, rank)
            entries = sorted(lp_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)
            topk_lps = [lp.logprob for _, lp in entries]
            ent = approx_entropy_from_topk(topk_lps, vocab_size)
            topk_list = [{
                "token": decode_token(tokenizer, tid),
                "prob": round(math.exp(lp.logprob), 6),
            } for tid, lp in entries]
            actual_lp = lp_dict.get(actual_tid)
            actual_prob = math.exp(actual_lp.logprob) if actual_lp is not None else 0.0

            token_str = decode_token(tokenizer, actual_tid)
            tokens_data.append({
                "pos": pos,
                "token": token_str,
                "entropy": round(ent, 4),
                "actual_prob": round(actual_prob, 6),
                "cum_char": cum_char,
                "topk": topk_list,
            })
            cum_char += len(token_str)

        ent_mean = sum(t["entropy"] for t in tokens_data) / max(1, len(tokens_data))
        print(f"  Q{prob.idx}: {len(tokens_data)} tokens, ent={ent_mean:.3f}, correct={correct}")

        problem_records.append({
            "idx": prob.idx,
            "ground_truth": prob.ground_truth,
            "response_text": resp_text,
            "response_len": len(resp_ids),
            "correct": correct,
            "extracted_answer": ans or "",
            "tokens": tokens_data,
        })

    # 释放 vLLM 显存
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return {"name": model_name, "problems": problem_records, "top_k_used": eff_top_k, "vocab_size": vocab_size}


# ─────────────────────────── main ───────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--n-problems", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=20, help="top-K logprobs（vLLM 上限 20）")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--models", nargs="*", default=None, help="只跑指定名称的模型（默认全部）")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    models = ALL_MODELS
    if args.models:
        wanted = set(args.models)
        models = [(n, p) for n, p in ALL_MODELS if n in wanted]
        missing = wanted - {n for n, _ in models}
        if missing:
            print(f"[warn] 未知模型名: {missing}", file=sys.stderr)

    problems = load_problems(args.parquet, args.n_problems)

    output_path = Path(args.output)
    # 支持断点续跑：若已有结果，复用 problems & 已完成模型
    all_results = {
        "problems": [{"idx": p.idx, "question": p.question, "ground_truth": p.ground_truth} for p in problems],
        "models": [],
        "meta": {
            "approx_entropy": True,
            "top_k": min(args.top_k, 20),
            "max_new": args.max_new,
            "n_problems": args.n_problems,
        },
    }
    if output_path.exists():
        try:
            prev = json.loads(output_path.read_text())
            if prev.get("problems") == all_results["problems"]:
                all_results["models"] = prev.get("models", [])
                done = {m["name"] for m in all_results["models"]}
                models = [(n, p) for n, p in models if n not in done]
                print(f"[resume] 跳过已完成: {sorted(done)}")
        except Exception as e:
            print(f"[warn] 旧结果文件读取失败，忽略: {e}")

    for name, path in models:
        print(f"\n{'='*60}\n模型: {name}\n{'='*60}")
        t0 = time.time()
        try:
            rec = eval_model(name, path, problems, args.max_new, args.top_k, args.gpu_mem_util)
            all_results["models"].append(rec)
            # 每跑完一个就落盘，避免后面崩了前面白跑
            output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
            print(f"  [{name}] 用时: {time.time()-t0:.0f}s, 已保存中间结果")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {name} 失败: {e}", file=sys.stderr)
            traceback.print_exc()
            print(f"  跳过 {name}，继续后面的模型")

    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\n完成，数据保存: {output_path}")


if __name__ == "__main__":
    main()
