"""
用 math_verify 对已有 traj 复核：筛出 verifier 仍为「不正确」的记录。

读取 dead_zone_phase_a.jsonl（路径可调），对 first_wrong_traj / wrong_trajs 重跑 is_correct，
去掉被旧判定误标成全错的题目。
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from recipe.RLSD.rlsd.verifier import is_correct


def main():
    input_path = "/data3/yyy/verl/data/rlsd/dead_zone_phase_a.jsonl"
    output_path = "/data3/yyy/verl/data/rlsd/dead_zone_verified.jsonl"

    print(f"[recheck] 读取: {input_path}")
    with open(input_path) as f:
        records = [json.loads(line) for line in f]
    print(f"[recheck] 共 {len(records)} 条记录")

    verified_negative = []
    flipped = []

    for rec in records:
        gt = rec["ground_truth"]
        all_trajs = []

        if rec.get("first_wrong_traj"):
            all_trajs.append(rec["first_wrong_traj"])
        for t in rec.get("wrong_trajs", []):
            if t and t not in all_trajs:
                all_trajs.append(t)

        any_correct = any(is_correct(traj, gt) for traj in all_trajs)

        if any_correct:
            flipped.append(rec)
        else:
            verified_negative.append(rec)

    print(f"\n[recheck] 结果:")
    print(f"  输入记录数:     {len(records)}")
    print(f"  翻转(实为正确): {len(flipped)}")
    print(f"  仍为不正确:    {len(verified_negative)}")

    print(f"\n[recheck] 翻转的题目示例 (前 10 条):")
    for rec in flipped[:10]:
        from recipe.RLSD.rlsd.verifier import extract_boxed_answer
        traj = rec.get("first_wrong_traj", "")
        extracted = extract_boxed_answer(traj) or "(none)"
        print(f"  idx={rec['index']}  gt={rec['ground_truth']!r}  pred={extracted!r}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in verified_negative:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[recheck] 复核沿用不正确记录已保存到: {output_path}")


if __name__ == "__main__":
    main()
