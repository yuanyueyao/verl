#!/usr/bin/env python3
"""
Custom SFT training for SFT baseline — aligned with SD experiment setup.

Usage:
    # 1.5B (single GPU or multi-GPU with accelerate)
    accelerate launch --num_processes=8 recipe/RLSD/sft_train.py \
        --model /data3/yyy/models/DeepSeek-R1-Distill-Qwen-1.5B \
        --data /data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet \
        --response_column COT_Reason \
        --output_dir /data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b \
        --total_steps 100 --lr 5e-6 --batch_size 64 \
        --eval_every 10

    # 7B (same but larger model)
    accelerate launch --num_processes=8 recipe/RLSD/sft_train.py \
        --model /data3/yyy/models/DeepSeek-R1-Distill-Qwen-7B \
        ... (same args)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# ── Eval imports ──
from vllm import LLM, SamplingParams

# ── Data ──────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """Tokenizes (prompt + response) with CE loss only on response tokens."""

    def __init__(self, parquet_path, tokenizer, max_len=24576,
                 prompt_column="problem", response_column="COT_Reason"):
        df = pd.read_parquet(parquet_path)
        self.prompts = df[prompt_column].tolist()
        self.responses = df[response_column].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        # Build chat-style messages
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # Fallback: plain text concatenation
            full = prompt + "\n\n" + response
        else:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            full = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

        # Tokenize
        tokenized = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]

        # Create labels: mask prompt portion
        prompt_messages = [
            {"role": "user", "content": prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        labels = [-100] * len(prompt_tokens) + input_ids[len(prompt_tokens):]

        # Pad if needed
        # Attention mask is all ones (no padding for single examples)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad batch to max length in batch."""
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids_padded = []
    labels_padded = []
    attention_mask = []

    for item in batch:
        L = item["input_ids"].size(0)
        pad_len = max_len - L
        input_ids_padded.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)])
        )
        labels_padded.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        mask = torch.cat([torch.ones(L), torch.zeros(pad_len)])
        attention_mask.append(mask)

    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
        "attention_mask": torch.stack(attention_mask),
    }


# ── Eval ──────────────────────────────────────────────────────

EPISTEMIC_SET = {"wait", "hmm", "perhaps", "maybe", "actually", "alternatively", "seems", "might", "likely", "check"}

def run_eval(model_path: str, eval_files: list[tuple[str, str]], step: int,
             output_dir: str, gpu_id: int = 0, max_samples_per_benchmark: int = 1000):
    """Evaluate with vLLM on multiple benchmarks. Returns metrics dict."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        llm = LLM(
            model=model_path,
            max_model_len=28672,
            gpu_memory_utilization=0.5,
            trust_remote_code=True,
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"[eval] vLLM init failed: {e}")
        return {}

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=16384,
        n=1,
    )
    # For AIME we need n=12 for acc@12
    aime_sampling = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=16384,
        n=12,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    all_results = []
    metrics_out = {}

    for bench_name, parquet_path in eval_files:
        df = pd.read_parquet(parquet_path)
        total = min(len(df), max_samples_per_benchmark)

        if bench_name.startswith("aime"):
            # AIME: need 12 samples per problem (acc@12)
            n_per_q = 12
            sp = aime_sampling
            micro_acc_name = "acc@12"
        else:
            n_per_q = 1
            sp = sampling_params
            micro_acc_name = "pass@1"

        # Build prompts
        prompts = []
        ground_truths = []
        indices = []

        for i in range(total):
            row = df.iloc[i]
            # Extract question text
            if "prompt" in df.columns and isinstance(row["prompt"], (list, np.ndarray)):
                # GSM8K-style
                q = row["prompt"][0]["content"]
            elif "question" in df.columns:
                q = row["question"]
            elif "problem" in df.columns:
                q = row["problem"]
            else:
                q = str(row.iloc[0])

            # Extract ground truth
            if "extra_info" in df.columns and isinstance(row["extra_info"], dict):
                gt = str(row["extra_info"].get("answer", row["extra_info"].get("ground_truth", "")))
            elif "reward_model" in df.columns and isinstance(row["reward_model"], dict):
                gt = str(row["reward_model"].get("ground_truth", ""))
            elif "ground_truth" in df.columns:
                gt = str(row["ground_truth"])
            else:
                gt = ""

            # Chat template
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": q}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = q + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

            for _ in range(n_per_q):
                prompts.append(prompt_text)
                ground_truths.append(gt)
                indices.append(i)

        # Generate
        outputs = llm.generate(prompts, sp)
        results_per_q = [[] for _ in range(total)]

        for j, output in enumerate(outputs):
            resp = output.outputs[0].text if output.outputs else ""
            # Extract \boxed{} answer
            extracted = _extract_boxed(resp)

            # Compare
            correct = _compare_answer(extracted, ground_truths[j])
            results_per_q[indices[j]].append({
                "response": resp,
                "extracted": extracted,
                "correct": correct,
            })

        # Aggregate
        if bench_name.startswith("aime"):
            # acc@12: fraction of problems with at least one correct trial
            n_correct_q = sum(1 for r in results_per_q if any(t["correct"] for t in r))
            acc = n_correct_q / total if total > 0 else 0
            # Micro average: fraction of all trials that are correct
            all_trials = [t for r in results_per_q for t in r]
            micro_acc = sum(1 for t in all_trials if t["correct"]) / len(all_trials) if all_trials else 0
            print(f"[eval] step={step} {bench_name} acc@12={acc:.3f} (micro={micro_acc:.3f})")
            metrics_out[f"val/{bench_name}/acc@12"] = acc
        else:
            n_correct = sum(1 for r in results_per_q if r[0]["correct"])
            acc = n_correct / total if total > 0 else 0
            print(f"[eval] step={step} {bench_name} pass@1={acc:.3f} ({n_correct}/{total})")
            metrics_out[f"val/{bench_name}/pass@1"] = acc

        # Save samples
        for idx, trials in enumerate(results_per_q):
            for t_idx, trial in enumerate(trials):
                all_results.append({
                    "step": step,
                    "benchmark": bench_name,
                    "problem_idx": idx,
                    "trial": t_idx,
                    "question": prompts[idx * n_per_q + t_idx][:500],
                    "ground_truth": ground_truths[idx * n_per_q + t_idx],
                    "response": trial["response"],
                    "extracted": trial["extracted"],
                    "correct": trial["correct"],
                })

    # Write samples
    os.makedirs(output_dir, exist_ok=True)
    samples_path = os.path.join(output_dir, "eval_samples_sft.jsonl")
    with open(samples_path, "a") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute macro mean
    acc_values = [v for k, v in metrics_out.items() if "pass@1" in k or "acc@12" in k]
    if acc_values:
        metrics_out["val/macro_mean"] = sum(acc_values) / len(acc_values)

    # Cleanup
    import gc
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_out


def _extract_boxed(text: str) -> str:
    """Extract content inside \boxed{} — robust to nesting."""
    if not text:
        return ""
    import re
    # Try simplest case first
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    brace = 0
    start = idx + len("\\boxed{")
    for i, c in enumerate(text[start:], start=start):
        if c == "{":
            brace += 1
        elif c == "}":
            if brace == 0:
                return text[start:i]
            brace -= 1
    return ""


def _compare_answer(extracted: str, ground_truth: str) -> bool:
    """Simple numeric/string comparison after normalization."""
    if not extracted or not ground_truth:
        return False

    def norm(s):
        s = s.strip().lower()
        # Remove LaTeX wrappers like \left, \right, \text, etc.
        for cmd in ["\\left", "\\right", "\\text", "\\displaystyle", "\\tfrac", "\\dfrac", "\\frac", "\\big", "\\Big", "\\bigg", "\\Bigg"]:
            s = s.replace(cmd, "")
        s = s.replace("{", "").replace("}", "")
        s = s.replace(" ", "").replace("\\", "")
        return s

    # Try numeric comparison
    try:
        ext_val = float(norm(extracted).replace(",", ""))
        gt_val = float(norm(ground_truth).replace(",", ""))
        return abs(ext_val - gt_val) < 1e-6
    except (ValueError, TypeError):
        pass

    # Try string comparison
    return norm(extracted) == norm(ground_truth)


# ── Training ──────────────────────────────────────────────────

def train(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation)

    is_main = accelerator.is_main_process

    if is_main:
        print(f"[SFT] model={args.model} data={args.data} steps={args.total_steps} lr={args.lr} bs={args.batch_size}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = SFTDataset(
        args.data, tokenizer, max_len=args.max_length,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
    )

    if is_main:
        print(f"[SFT] dataset size: {len(dataset)}")

    # Use DistributedSampler so each rank gets a different slice
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    if is_main:
        print("[SFT] loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.train()

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = args.total_steps
    warmup_steps = args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Accelerator prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if is_main:
        print(f"[SFT] training {total_steps} steps with {accelerator.num_processes} GPUs")
        print(f"      micro_batch={args.micro_batch_size} grad_accum={args.gradient_accumulation}")
        print(f"      effective_batch={args.batch_size}")

    # Training loop
    global_step = 0
    data_iter = iter(dataloader)
    eval_metrics_history = []

    while global_step < total_steps:
        model.train()

        # Fetch next batch; restart iterator if exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(global_step)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1

            if is_main:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step={global_step}/{total_steps} loss={loss.item():.4f} lr={lr_now:.2e}")

            # Periodic eval
            if global_step % args.eval_every == 0 and global_step > 0:
                if is_main:
                    print(f"\n[eval] step={global_step} saving checkpoint...")
                    # Save checkpoint
                    ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt_dir)

                    # Run eval
                    eval_files = [
                        ("aime_2024", "/data3/yyy/verl/data/math/val_aime_2024.parquet"),
                        ("aime_2025", "/data3/yyy/verl/data/math/val_aime_2025.parquet"),
                        ("MATH-500", "/data3/yyy/verl/data/math/val_MATH-500.parquet"),
                    ]
                    # Add GSM8K if available
                    gsm_path = "/data3/yyy/verl/data/gsm8k/test.parquet"
                    if os.path.exists(gsm_path):
                        eval_files.append(("GSM8K", gsm_path))

                    print(f"[eval] running eval on {len(eval_files)} benchmarks...")
                    # Need to sync before eval (free GPU memory)
                    metrics = run_eval(
                        ckpt_dir, eval_files, global_step,
                        args.output_dir, gpu_id=0,
                        max_samples_per_benchmark=args.eval_max_samples,
                    )
                    eval_metrics_history.append({"step": global_step, **metrics})

                    # Save metrics
                    metrics_path = os.path.join(args.output_dir, "eval_metrics.jsonl")
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps({"step": global_step, **metrics}) + "\n")

                    print(f"[eval] done. metrics: {json.dumps({k: round(v,3) for k,v in metrics.items()})}")

    # Final checkpoint
    if is_main:
        ckpt_dir = os.path.join(args.output_dir, "final")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"[SFT] training complete. final checkpoint: {ckpt_dir}")

    accelerator.end_training()


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Custom SFT training for baseline")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--response_column", type=str, default="COT_Reason")
    parser.add_argument("--prompt_column", type=str, default="problem")
    parser.add_argument("--total_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=24576)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_max_samples", type=int, default=1000)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Auto-compute gradient accumulation
    if args.gradient_accumulation is None:
        import torch.distributed as dist
        ngpus = dist.get_world_size() if dist.is_initialized() else 1
        # effective_batch = ngpus * micro_batch * grad_accum
        denom = ngpus * args.micro_batch_size
        args.gradient_accumulation = max(1, args.batch_size // denom)
        effective_batch = ngpus * args.micro_batch_size * args.gradient_accumulation
        if effective_batch != args.batch_size:
            print(f"[warn] effective batch {effective_batch} != target {args.batch_size}, adjusting")
            # Round to nearest divisible
            args.batch_size = effective_batch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
