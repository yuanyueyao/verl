#!/usr/bin/env python3
"""
Custom SFT training for baseline — train only (eval runs separately).

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python recipe/RLSD/sft_train.py \
        --model /data3/yyy/models/DeepSeek-R1-Distill-Qwen-1.5B \
        --data /data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet \
        --output_dir /data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b \
        --eval_every 10

    # 8-GPU DDP
    torchrun --standalone --nnodes=1 --nproc_per_node=8 recipe/RLSD/sft_train.py \
        --model /data3/yyy/models/DeepSeek-R1-Distill-Qwen-1.5B \
        --data /data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet \
        --output_dir /data3/yyy/verl/checkpoints/sft_exp_ds_qwen1.5b \
        --eval_every 10
"""

import argparse
import contextlib
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

from recipe.RLSD.rlsd.prompt import build_student_messages


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
        question = self.prompts[idx]
        response = self.responses[idx]

        # Use build_student_messages to match eval prompt format exactly
        msgs = build_student_messages(question)
        msgs.append({"role": "assistant", "content": response})
        full = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

        tokenized = self.tokenizer(full, truncation=True, max_length=self.max_len,
                                   padding=False, return_tensors=None)
        input_ids = tokenized["input_ids"]

        # Mask prompt portion: everything before the assistant response
        prompt_msgs = build_student_messages(question)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_tokens), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids_padded, labels_padded, masks = [], [], []
        for item in batch:
            L = item["input_ids"].size(0)
            pad_len = max_len - L
            input_ids_padded.append(
                torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )
            labels_padded.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            masks.append(torch.cat([torch.ones(L, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids_padded),
            "labels": torch.stack(labels_padded),
            "attention_mask": torch.stack(masks),
        }

    return collate_fn


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return distributed, rank, local_rank, world_size


def is_main_process(rank):
    return rank == 0


def main_print(rank, *args, **kwargs):
    if is_main_process(rank):
        print(*args, **kwargs)


def run_vllm_eval(args, rank, model, tokenizer, step):
    if args.eval_every <= 0 or step % args.eval_every != 0:
        return
    if not is_main_process(rank):
        return

    eval_output_dir = args.eval_output_dir or args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)
    tmp_root = args.tmp_dir or tempfile.gettempdir()
    tmp_ckpt = tempfile.mkdtemp(prefix=f"sft_step_{step}_", dir=tmp_root)
    unwrapped = model.module if isinstance(model, DDP) else model

    main_print(rank, f"[SFT] eval step={step}: writing temporary weights to {tmp_ckpt}")
    unwrapped.save_pretrained(tmp_ckpt, safe_serialization=True)
    tokenizer.save_pretrained(tmp_ckpt)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "sft_eval.py"),
        "--model", tmp_ckpt,
        "--output_dir", eval_output_dir,
        "--step", str(step),
        "--max_samples", str(args.eval_max_samples),
    ]
    env = os.environ.copy()
    if args.eval_gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.eval_gpu

    try:
        main_print(rank, f"[SFT] eval step={step}: vLLM on CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
        subprocess.run(cmd, check=True, env=env)
    finally:
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
        main_print(rank, f"[SFT] eval step={step}: removed temporary weights")


def train(args):
    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    main_print(rank, f"[SFT] model={args.model} steps={args.total_steps} lr={args.lr} global_bs={args.batch_size} world_size={world_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SFTDataset(args.data, tokenizer, max_len=args.max_length,
                         prompt_column=args.prompt_column,
                         response_column=args.response_column)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    dataloader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=(sampler is None),
                            sampler=sampler, collate_fn=make_collate_fn(tokenizer.pad_token_id),
                            num_workers=0, drop_last=True)

    main_print(rank, f"[SFT] dataset={len(dataset)} samples, {len(dataloader)} batches/epoch/rank")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    denom = args.micro_batch_size * world_size
    grad_accum = max(1, math.ceil(args.batch_size / denom))
    effective_bs = args.micro_batch_size * grad_accum * world_size
    main_print(rank, f"[SFT] micro_batch_per_gpu={args.micro_batch_size} grad_accum={grad_accum} effective_bs={effective_bs}")
    if effective_bs != args.batch_size:
        main_print(rank, f"[SFT] warning: requested batch_size={args.batch_size}, using effective_bs={effective_bs}")

    data_iter = iter(dataloader)
    effective_step = 0
    micro_step = 0
    epoch = 0
    optimizer.zero_grad()

    while effective_step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        sync_grad = (micro_step + 1) % grad_accum == 0
        sync_context = model.no_sync() if distributed and not sync_grad else contextlib.nullcontext()
        with sync_context:
            outputs = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True),
            )
            loss = outputs.loss / grad_accum
            loss.backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_step += 1

            main_print(rank, f"  step={effective_step}/{args.total_steps} loss={loss.item() * grad_accum:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

            if args.save_every > 0 and effective_step % args.save_every == 0:
                if distributed:
                    dist.barrier()
                if is_main_process(rank):
                    ckpt_dir = os.path.join(args.output_dir, f"step_{effective_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    unwrapped = model.module if isinstance(model, DDP) else model
                    main_print(rank, f"[SFT] saving checkpoint to {ckpt_dir}")
                    unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt_dir)
                if distributed:
                    dist.barrier()

            if args.eval_every > 0 and effective_step % args.eval_every == 0:
                if distributed:
                    dist.barrier()
                run_vllm_eval(args, rank, model, tokenizer, effective_step)
                if distributed:
                    dist.barrier()

    main_print(rank, "[SFT] done. checkpoints were not saved.")
    if args.save_final_dir:
        if distributed:
            dist.barrier()
        if is_main_process(rank):
            os.makedirs(args.save_final_dir, exist_ok=True)
            unwrapped = model.module if isinstance(model, DDP) else model
            main_print(rank, f"[SFT] writing temporary final weights to {args.save_final_dir}")
            unwrapped.save_pretrained(args.save_final_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.save_final_dir)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--response_column", type=str, default="COT_Reason")
    parser.add_argument("--prompt_column", type=str, default="problem")
    parser.add_argument("--total_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=24576)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_max_samples", type=int, default=1000)
    parser.add_argument("--eval_output_dir", type=str, default=None)
    parser.add_argument("--eval_gpu", type=str, default=None)
    parser.add_argument("--tmp_dir", type=str, default=None)
    parser.add_argument("--save_final_dir", type=str, default=None)

    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train(args)


if __name__ == "__main__":
    main()
