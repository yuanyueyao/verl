"""
Zero-RL 奖励函数：用 math_verify 库做数学等价判题。
替代默认的 math_dapo.compute_score（字符串匹配）。
"""
import sys
import os

VERL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if VERL_ROOT not in sys.path:
    sys.path.insert(0, VERL_ROOT)

from recipe.RLSD.rlsd.verifier import is_correct


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    correct = is_correct(solution_str, str(ground_truth))
    return {"score": 1.0 if correct else -1.0, "acc": float(correct)}
