#!/usr/bin/env python3
"""
将 dapo-math-17k.parquet 的 prompt 从 "Answer: $Answer" 改为 boxed 格式。
输出到同目录的 dapo-math-17k-boxed.parquet。
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

DATA_DIR = Path("/data3/yyy/verl/data")
INPUT = DATA_DIR / "dapo-math-17k.parquet"
OUTPUT = DATA_DIR / "dapo-math-17k-boxed.parquet"

BOXED_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def convert_prompt(old_prompt):
    if not isinstance(old_prompt, (list, np.ndarray)) or len(old_prompt) == 0:
        return old_prompt

    content = old_prompt[0].get("content", "") if hasattr(old_prompt[0], "get") else old_prompt[0]["content"]
    match = re.search(r"problem\.\n\n(.+?)(?:\n\nRemember to put|$)", content, re.DOTALL)
    if not match:
        return old_prompt

    problem = match.group(1).strip()
    new_content = BOXED_TEMPLATE.format(problem=problem)
    return [{"content": new_content, "role": "user"}]


def main():
    print(f"Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    print(f"Rows: {len(df)}")

    print("Converting prompts...")
    converted_prompts = []
    for i in range(len(df)):
        if i % 200000 == 0:
            print(f"  {i}/{len(df)}")
        converted_prompts.append(convert_prompt(df["prompt"].iloc[i]))

    print("Creating DataFrame...")
    df_new = pd.DataFrame({
        "data_source": df["data_source"],
        "prompt": converted_prompts,
        "ability": df["ability"],
        "reward_model": df["reward_model"],
        "extra_info": df["extra_info"],
    })

    print(f"Saving {OUTPUT}...")
    df_new.to_parquet(OUTPUT, index=False)

    df_check = pd.read_parquet(OUTPUT)
    sample = df_check["prompt"].iloc[0][0]["content"]
    print(f"\nDone! {len(df_check)} rows")
    print(f"Has 'boxed': {'boxed' in sample}")
    print(f"No old format: {'Answer: $Answer' not in sample}")
    print(sample[:300])


if __name__ == "__main__":
    main()
