#!/usr/bin/env python3
"""
Eval saved SFT checkpoint on AIME24/25, MATH-500, GSM8K.

Usage:
    python recipe/RLSD/sft_eval.py --model /path/to/checkpoint --output_dir /path/to/results
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recipe.RLSD.rlsd.prompt import build_student_messages, question_from_verl_prompt
from recipe.RLSD.rlsd.verifier import extract_boxed_answer, is_correct


EPISTEMIC_SET = {"wait", "hmm", "perhaps", "maybe", "actually", "alternatively", "seems", "might", "likely", "check"}


def get_question(row, columns):
    if "prompt" in columns and isinstance(row["prompt"], (list, np.ndarray)):
        return question_from_verl_prompt(row["prompt"])
    if "question" in columns:
        return row["question"]
    if "problem" in columns:
        return row["problem"]
    return str(row.iloc[0])


def get_ground_truth(row, columns):
    if "reward_model" in columns and isinstance(row["reward_model"], dict):
        gt = row["reward_model"].get("ground_truth", "")
        if gt:
            return str(gt)
    if "ground_truth" in columns:
        gt = row["ground_truth"]
        if gt:
            return str(gt)
    if "extra_info" in columns and isinstance(row["extra_info"], dict):
        gt = row["extra_info"].get("ground_truth", "")
        if gt:
            return str(gt)
        answer = str(row["extra_info"].get("answer", ""))
        if "####" in answer:
            return answer.split("####")[-1].strip()
        return answer
    return ""


def metric_name(bench_name):
    return "avg@12" if bench_name.startswith("aime") else "pass@1"


def aggregate_shards(output_dir: str, step: int, num_shards: int):
    grouped = {}
    combined = []
    for shard_id in range(num_shards):
        path = os.path.join(output_dir, f"eval_samples_sft_step_{step}_shard_{shard_id}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                combined.append(item)
                key = (item["benchmark"], item["problem_idx"])
                grouped.setdefault(key, []).append(item)

    by_bench = {}
    for (bench_name, problem_idx), trials in grouped.items():
        by_bench.setdefault(bench_name, {})[problem_idx] = trials

    metrics = {"step": step}
    for bench_name, problems in sorted(by_bench.items()):
        if bench_name.startswith("aime"):
            frac_sum = sum(sum(1 for t in trials if t["correct"]) / len(trials) for trials in problems.values() if trials)
            n_correct = sum(1 for trials in problems.values() for t in trials if t["correct"])
            n_trials = sum(len(trials) for trials in problems.values())
            acc = frac_sum / len(problems) if problems else 0
            metrics[f"val/{bench_name}/avg@12"] = acc
            metrics[f"val/{bench_name}/n_correct_trials"] = float(n_correct)
            metrics[f"val/{bench_name}/n_total_trials"] = float(n_trials)
            print(f"[eval] step={step} {bench_name} avg@12={acc:.3f} ({n_correct}/{n_trials} trials)")
        else:
            n_correct = sum(1 for trials in problems.values() if trials and trials[0]["correct"])
            acc = n_correct / len(problems) if problems else 0
            metrics[f"val/{bench_name}/pass@1"] = acc
            print(f"[eval] step={step} {bench_name} pass@1={acc:.3f} ({n_correct}/{len(problems)})")

    acc_values = [v for k, v in metrics.items() if "pass@1" in k or "avg@12" in k]
    if acc_values:
        metrics["val/macro_mean"] = sum(acc_values) / len(acc_values)

    combined_path = os.path.join(output_dir, f"eval_samples_sft_step_{step}.jsonl")
    with open(combined_path, "w") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "eval_metrics.jsonl"), "a") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    print(f"[eval] aggregate done. metrics: {json.dumps({k: round(v,3) for k,v in metrics.items()})}")
    return metrics


def run_eval(model_path: str, output_dir: str, step: int = 0, max_samples: int = 64,
             shard_id: int = 0, num_shards: int = 1, max_tokens: int = 16384,
             include_gsm8k: bool = False, gsm8k_max_samples: int | None = None):
    from vllm import LLM, SamplingParams

    eval_files = [
        ("aime_2024", "/data3/yyy/verl/data/math/val_aime_2024.parquet"),
        ("aime_2025", "/data3/yyy/verl/data/math/val_aime_2025.parquet"),
        ("MATH-500", "/data3/yyy/verl/data/math/val_MATH-500.parquet"),
    ]
    gsm_path = "/data3/yyy/verl/data/gsm8k/test.parquet"
    if include_gsm8k and os.path.exists(gsm_path):
        eval_files.append(("GSM8K", gsm_path))

    print(f"[eval] model={model_path} benchmarks={[f[0] for f in eval_files]}")

    llm = LLM(model=model_path, max_model_len=28672, gpu_memory_utilization=0.5,
              trust_remote_code=True, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sp_pass1 = SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=max_tokens, n=1)
    # We explicitly expand AIME prompts 12 times below, matching RLSD evaluation.
    # Keep n=1 here to avoid generating 12x redundant completions per expanded prompt.
    sp_aime = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=max_tokens, n=1)

    all_results = []
    metrics = {}

    for bench_name, parquet_path in eval_files:
        df = pd.read_parquet(parquet_path)
        bench_max_samples = gsm8k_max_samples if bench_name == "GSM8K" and gsm8k_max_samples is not None else max_samples
        total = len(df) if bench_max_samples < 0 else min(len(df), bench_max_samples)
        is_aime = bench_name.startswith("aime")
        n_per_q = 12 if is_aime else 1
        sp = sp_aime if is_aime else sp_pass1

        selected = [i for i in range(total) if i % num_shards == shard_id]
        prompts, gts, indices = [], [], []
        for i in selected:
            row = df.iloc[i]
            q = get_question(row, df.columns)
            gt = get_ground_truth(row, df.columns)

            msgs = build_student_messages(q)
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            for _ in range(n_per_q):
                prompts.append(prompt_text)
                gts.append(gt)
                indices.append(i)

        outputs = llm.generate(prompts, sp)
        results_per_q = {i: [] for i in selected}

        for j, out in enumerate(outputs):
            resp = out.outputs[0].text if out.outputs else ""
            extracted = extract_boxed_answer(resp) or ""
            correct = is_correct(resp, gts[j])
            results_per_q[indices[j]].append({"response": resp, "extracted": extracted, "correct": correct})

        if is_aime:
            n_correct_q = sum(1 for r in results_per_q.values() for t in r if t["correct"])
            acc = sum(sum(1 for t in r if t["correct"]) / len(r) for r in results_per_q.values() if r) / len(selected) if selected else 0
            all_trials = [t for r in results_per_q.values() for t in r]
            micro = sum(1 for t in all_trials if t["correct"]) / len(all_trials) if all_trials else 0
            print(f"[eval] step={step} shard={shard_id}/{num_shards} {bench_name} avg@12={acc:.3f} (micro={micro:.3f} n_correct_trials={n_correct_q}/{len(all_trials)})")
            metrics[f"val/{bench_name}/avg@12"] = acc
        else:
            n_correct = sum(1 for r in results_per_q.values() if r and r[0]["correct"])
            acc = n_correct / len(selected) if selected else 0
            print(f"[eval] step={step} shard={shard_id}/{num_shards} {bench_name} pass@1={acc:.3f} ({n_correct}/{len(selected)})")
            metrics[f"val/{bench_name}/pass@1"] = acc

        gt_by_problem = {}
        for idx, gt in zip(indices, gts):
            gt_by_problem.setdefault(idx, gt)
        for idx, trials in results_per_q.items():
            for t_idx, trial in enumerate(trials):
                all_results.append({
                    "step": step, "benchmark": bench_name, "problem_idx": idx,
                    "trial": t_idx, "ground_truth": gt_by_problem.get(idx, ""),
                    "response": trial["response"], "extracted": trial["extracted"],
                    "correct": trial["correct"],
                })

    os.makedirs(output_dir, exist_ok=True)
    if num_shards > 1:
        sample_name = f"eval_samples_sft_step_{step}_shard_{shard_id}.jsonl"
        metrics_name = f"eval_metrics_step_{step}_shard_{shard_id}.json"
    else:
        sample_name = "eval_samples_sft.jsonl"
        metrics_name = "eval_metrics.json"

    with open(os.path.join(output_dir, sample_name), "w" if num_shards > 1 else "a") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc_values = [v for k, v in metrics.items() if "pass@1" in k or "avg@12" in k]
    if acc_values:
        metrics["val/macro_mean"] = sum(acc_values) / len(acc_values)

    metrics["step"] = step
    metrics["shard_id"] = shard_id
    metrics["num_shards"] = num_shards
    with open(os.path.join(output_dir, metrics_name), "w") as f:
        json.dump(metrics, f, indent=2)
    if num_shards == 1:
        with open(os.path.join(output_dir, "eval_metrics.jsonl"), "a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    print(f"[eval] done. metrics: {json.dumps({k: round(v,3) for k,v in metrics.items()})}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--include_gsm8k", action="store_true")
    parser.add_argument("--gsm8k_max_samples", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_shards(args.output_dir, args.step, args.num_shards)
        return

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_eval(args.model, args.output_dir, step=args.step, max_samples=args.max_samples,
             shard_id=args.shard_id, num_shards=args.num_shards,
             max_tokens=args.max_tokens, include_gsm8k=args.include_gsm8k,
             gsm8k_max_samples=args.gsm8k_max_samples)


if __name__ == "__main__":
    main()
