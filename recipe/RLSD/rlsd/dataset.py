"""
MRSD 问题池。

采样始终覆盖池内全部题目（均匀随机）；不再维护「活跃 / 毕业」子集或训练期将题目移出池子。
仍可记录 per-problem 训练统计（对错次数、错误轨迹等）供诊断；序列化字段与旧 checkpoint 兼容。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class MRSDProblem:
    """训练题池中的单题状态（题干、可验证答案、可选参考解、训练中累计统计等）。"""

    __slots__ = (
        "index",
        "question",
        "ground_truth",
        "reference_solution",    # 数据集中参考答案（推导过程）；仅 teacher 特权 prompt 使用
        "difficulty",
        "topic",
        "wrong_trajs",         # 诊断阶段收集的错误轨迹（初始 seed，后续动态更新）
        "n_correct_at_train",  # 训练累计答对（诊断用）
        "n_total_at_train",    # 训练累计采样次数
        "graduated",           # 遗留字段；不参与采样，load 后恒为 False
    )

    def __init__(
        self,
        index: int,
        question: str,
        ground_truth: str,
        difficulty: float = 5.0,
        topic: str = "",
        wrong_trajs: Optional[list[str]] = None,
        reference_solution: str = "",
    ):
        self.index = index
        self.question = question
        self.ground_truth = ground_truth
        self.reference_solution = reference_solution
        self.difficulty = difficulty
        self.topic = topic
        self.wrong_trajs = wrong_trajs or []
        self.n_correct_at_train = 0
        self.n_total_at_train = 0
        self.graduated = False

    def update_stats(self, n_correct: int, n_total: int) -> None:
        self.n_correct_at_train += n_correct
        self.n_total_at_train += n_total

    @property
    def pass_at_1_estimate(self) -> float:
        if self.n_total_at_train == 0:
            return 0.0
        return self.n_correct_at_train / self.n_total_at_train

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "reference_solution": self.reference_solution,
            "difficulty": self.difficulty,
            "topic": self.topic,
            "wrong_trajs": self.wrong_trajs[:2],  # 只序列化前2条
            "n_correct_at_train": self.n_correct_at_train,
            "n_total_at_train": self.n_total_at_train,
            "graduated": self.graduated,
        }


class MRSDDataset:
    """
    训练题池：`sample_batch` 始终从全部题目中抽取；`maybe_graduate_problems` 为空操作（接口保留）。
    """

    def __init__(
        self,
        problems: list[MRSDProblem],
        seed: int = 42,
        graduation_pass_at_k: int = 4,           # 已废弃，仅为旧配置/序列化兼容
        graduation_interval: int = 100,
        graduation_threshold: float = 0.0,
    ):
        self.problems = {p.index: p for p in problems}
        self._rng = random.Random(seed)
        # JSON 字段名仍为 active_indices：语义为「全体可采样 index」，sorted 以保证稳定顺序
        self.active_indices = sorted(self.problems.keys())
        self.graduated_indices: list[int] = []
        self.graduation_pass_at_k = graduation_pass_at_k
        self.graduation_interval = graduation_interval
        self.graduation_threshold = graduation_threshold

    # ──────────────────────────────────────────────────────────────────
    # 类方法：从诊断结果文件构建数据集
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_pass_at_k_results(
        cls,
        pass_at_k_jsonl: str,
        type_b_only: bool = True,
        **kwargs,
    ) -> "MRSDDataset":
        """
        从 run_pass_at_k.py 生成的 jsonl 构建数据集。
        type_b_only=True 时只载入 ``is_dead_zone`` 为真的记录（通常为 pass@k 诊断里「k 次无一次判对」子集；
        字段名沿用 jsonl，并非运行时「活跃池」语义）。
        """
        problems = []
        with open(pass_at_k_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if type_b_only and not rec.get("is_dead_zone", False):
                    continue
                problems.append(
                    MRSDProblem(
                        index=rec["index"],
                        question=rec["question"],
                        ground_truth=rec["ground_truth"],
                        difficulty=float(rec.get("difficulty", 5.0)),
                        topic=rec.get("topic", ""),
                        wrong_trajs=rec.get("wrong_trajs", []),
                        reference_solution=str(rec.get("reference_solution") or rec.get("solution") or ""),
                    )
                )
        print(f"[MRSDDataset] 从 {pass_at_k_jsonl} 加载 {len(problems)} 道题目")
        return cls(problems=problems, **kwargs)

    @classmethod
    def from_context_ab_results(
        cls,
        type_b_jsonl: str,
        **kwargs,
    ) -> "MRSDDataset":
        """
        从 run_context_ab_test.py 生成的 type_b_problems.jsonl 构建数据集。
        包含带初始错误轨迹的候选题目（见 diagnostic 流水线输出格式）。
        """
        problems = []
        with open(type_b_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                problems.append(
                    MRSDProblem(
                        index=rec["index"],
                        question=rec["question"],
                        ground_truth=rec["ground_truth"],
                        difficulty=float(rec.get("difficulty", 5.0)),
                        topic=rec.get("topic", ""),
                        wrong_trajs=rec.get("wrong_trajs", []),
                        reference_solution=str(rec.get("reference_solution") or rec.get("solution") or ""),
                    )
                )
        print(f"[MRSDDataset] 从 {type_b_jsonl} 加载 {len(problems)} 道题目（含错误轨迹）")
        return cls(problems=problems, **kwargs)

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        **kwargs,
    ) -> "MRSDDataset":
        """从 OpenThoughts 导出的 parquet 构建题目列表（列：problem、Answer、solution）。"""
        df = pd.read_parquet(parquet_path)
        for col in ("problem", "Answer", "solution"):
            if col not in df.columns:
                raise ValueError(
                    f"期望 OpenThoughts parquet 含列 {col!r}，实际列：{list(df.columns)}"
                )
        problems = [
            MRSDProblem(
                index=int(i),
                question=str(row["problem"]),
                ground_truth=str(row["Answer"]),
                difficulty=5.0,
                topic="",
                reference_solution=str(row["solution"]),
            )
            for i, row in df.iterrows()
        ]
        print(f"[MRSDDataset] 从 {parquet_path} 加载 {len(problems)} 道题目")
        return cls(problems=problems, **kwargs)

    # ──────────────────────────────────────────────────────────────────
    # 访问接口
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.problems)

    def sample_batch(self, n: int, replace: bool = False) -> list[MRSDProblem]:
        """从全池随机抽 n 题（不放回；n 大于池大则 clamp）。"""
        pool = self.active_indices
        if len(pool) == 0:
            return []
        n = min(n, len(pool))
        if replace:
            indices = self._rng.choices(pool, k=n)
        else:
            indices = self._rng.sample(pool, n)
        return [self.problems[i] for i in indices]

    def get_problem(self, index: int) -> Optional[MRSDProblem]:
        return self.problems.get(index)

    @property
    def n_active(self) -> int:
        """与 ``len(dataset)`` 相同；保留原名以免外部脚本 breakage。"""
        return len(self.problems)

    @property
    def n_graduated(self) -> int:
        """恒为 0；保留接口兼容。"""
        return 0

    # ──────────────────────────────────────────────────────────────────
    # 状态更新
    # ──────────────────────────────────────────────────────────────────

    def update_problem_stats(
        self,
        index: int,
        new_wrong_trajs: Optional[list[str]] = None,
        n_correct: int = 0,
        n_total: int = 0,
    ) -> None:
        """
        训练后更新题目状态：
          - 追加新的错误轨迹（on-policy）
          - 更新正确率统计
        """
        prob = self.problems.get(index)
        if prob is None:
            return
        if new_wrong_trajs:
            # 保留最新的 4 条错误轨迹（on-policy，丢弃旧的 off-policy 轨迹）
            prob.wrong_trajs = (new_wrong_trajs + prob.wrong_trajs)[:4]
        prob.update_stats(n_correct, n_total)

    def maybe_graduate_problems(self, force: bool = False) -> list[int]:
        """已停用：不进行毕业、不改变池子。"""
        return []

    # ──────────────────────────────────────────────────────────────────
    # 序列化（checkpoint 用）
    # ──────────────────────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        state = {
            "active_indices": sorted(self.problems.keys()),
            "graduated_indices": [],
            "problems": {str(k): v.to_dict() for k, v in self.problems.items()},
        }
        with open(path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, path: str) -> None:
        with open(path) as f:
            state = json.load(f)
        # 恢复统计数据后固定全池训练，忽略旧 checkpoint 中的毕业/活跃划分
        self.graduated_indices = []
        self.active_indices = sorted(self.problems.keys())
        for idx_str, pdata in state["problems"].items():
            idx = int(idx_str)
            if idx in self.problems:
                p = self.problems[idx]
                p.n_correct_at_train = pdata.get("n_correct_at_train", 0)
                p.n_total_at_train = pdata.get("n_total_at_train", 0)
                p.graduated = False
                if pdata.get("wrong_trajs"):
                    p.wrong_trajs = pdata["wrong_trajs"]

    # ──────────────────────────────────────────────────────────────────
    # 调试
    # ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        from collections import Counter
        topics = Counter(p.topic for p in self.problems.values())
        n = len(self.problems)
        return {
            "n_total": n,
            "n_problems": n,
            "topic_distribution": dict(topics.most_common(10)),
        }
