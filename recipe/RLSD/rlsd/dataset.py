"""
RLSD 问题数据集。

从 parquet 加载全量题目列表，随机采样供训练使用。
"""

from __future__ import annotations

import random

import pandas as pd


class RLSDProblem:
    """训练题池中的单题（题干、可验证答案、可选参考解）。"""

    __slots__ = (
        "index",
        "question",
        "ground_truth",
        "reference_solution",
    )

    def __init__(
        self,
        index: int,
        question: str,
        ground_truth: str,
        reference_solution: str = "",
    ):
        self.index = index
        self.question = question
        self.ground_truth = ground_truth
        self.reference_solution = reference_solution


class RLSDDataset:
    """全量训练题池，均匀随机采样。"""

    def __init__(self, problems: list[RLSDProblem], seed: int = 42):
        self.problems = problems
        self._rng = random.Random(seed)

    @classmethod
    def from_parquet(cls, parquet_path: str, seed: int = 42) -> "RLSDDataset":
        """从 parquet 构建题目列表（列：problem、Answer、solution）。"""
        df = pd.read_parquet(parquet_path)
        for col in ("problem", "Answer"):
            if col not in df.columns:
                raise ValueError(f"parquet 缺少列 {col!r}，实际列：{list(df.columns)}")
        problems = [
            RLSDProblem(
                index=int(i),
                question=str(row["problem"]),
                ground_truth=str(row["Answer"]),
                reference_solution=str(row.get("solution", "")),
            )
            for i, row in df.iterrows()
        ]
        print(f"[RLSDDataset] 从 {parquet_path} 加载 {len(problems)} 道题目")
        return cls(problems=problems, seed=seed)

    def __len__(self) -> int:
        return len(self.problems)

    def sample_batch(self, n: int) -> list[RLSDProblem]:
        """从全池随机抽 n 题（不放回）。"""
        n = min(n, len(self.problems))
        indices = self._rng.sample(range(len(self.problems)), n)
        return [self.problems[i] for i in indices]
