"""
把 ``/data3/yyy/verl/data/math`` 下 HF ``save_to_disk`` 目录转为 verl 评测 parquet
（列：prompt、reward_model），供 MRSDTrainer._evaluate 使用。

用法（仓库根目录）::

    PYTHONPATH=/data3/yyy/verl python recipe/RLSD/data/export_math_val_parquets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe.RLSD.rlsd.prompt import build_student_messages


MATH_ROOT = Path("/data3/yyy/verl/data/math")


def _verl_row(prompt_messages: list, gt, source_tag: str, idx: int) -> dict:
    return {
        "data_source": source_tag,
        "prompt": prompt_messages,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(gt)},
        "extra_info": {"index": idx, "benchmark": source_tag},
    }


def main() -> None:
    math_dir = MATH_ROOT

    ds = load_from_disk(str(math_dir / "MATH-500"))
    split = ds["test"]
    rows = [_verl_row(build_student_messages(split[i]["problem"]), split[i]["answer"], "MATH-500", i) for i in range(len(split))]
    pd.DataFrame(rows).to_parquet(math_dir / "val_MATH-500.parquet", index=False)
    print(f"val_MATH-500.parquet  rows={len(rows)}")

    ds = load_from_disk(str(math_dir / "aime_2024"))
    split = ds["train"]
    rows = [_verl_row(build_student_messages(split[i]["problem"]), split[i]["answer"], "aime_2024", i) for i in range(len(split))]
    pd.DataFrame(rows).to_parquet(math_dir / "val_aime_2024.parquet", index=False)
    print(f"val_aime_2024.parquet  rows={len(rows)}")

    ds = load_from_disk(str(math_dir / "aime_2025"))
    split = ds["test"]
    rows = [_verl_row(build_student_messages(split[i]["question"]), split[i]["answer"], "aime_2025", i) for i in range(len(split))]
    pd.DataFrame(rows).to_parquet(math_dir / "val_aime_2025.parquet", index=False)
    print(f"val_aime_2025.parquet  rows={len(rows)}")


if __name__ == "__main__":
    main()
