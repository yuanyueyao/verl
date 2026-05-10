"""
实验：给 Teacher 完整 reference solution，观察是否能内化为自己风格的推理。

对比：
  A - Student prompt（无特权信息）
  B - Teacher privileged prompt（仅 GT 数字答案）
  C - Teacher privileged prompt（完整 reference solution）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlsd.prompt import build_student_messages, build_teacher_privileged_messages
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

MODEL_PATH = "/data3/yyy/models/Qwen2.5-3B-Instruct"
PARQUET_PATH = "/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet"
PROBLEM_IDX = 2    # xyz divisible by 10, answer=72, student says 91
N_SAMPLES = 2
MAX_NEW_TOKENS = 1200

# ── 加载数据 ──────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
row = df.iloc[PROBLEM_IDX]
QUESTION = row["problem"]
GROUND_TRUTH = str(row["Answer"])
REFERENCE_SOLUTION = row["solution"]

print("=" * 70)
print(f"题目:\n{QUESTION}\n")
print(f"标准答案: {GROUND_TRUTH}")
print(f"Reference solution 长度: {len(REFERENCE_SOLUTION)} 字符")
print(f"Reference solution:\n{REFERENCE_SOLUTION}\n")

# ── 加载模型 ──────────────────────────────────────────────────────────────
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()
print("模型加载完成\n")


def generate(messages, label, n=N_SAMPLES):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    print(f"\n{'=' * 70}")
    print(f"【{label}】")
    print(f"Prompt token 数: {input_len}")

    for i in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(i > 0),
                temperature=0.7 if i > 0 else 1.0,
                top_p=0.9 if i > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        correct = GROUND_TRUTH in response
        copied = REFERENCE_SOLUTION[:80].strip() in response  # 检测是否直接抄录

        tag = " [✓ 含正确答案]" if correct else " [✗ 未含正确答案]"
        if copied:
            tag += " [⚠ 疑似直接抄录 reference]"

        print(f"\n[样本 {i+1}] {'greedy' if i==0 else 'top-p'}{tag}")
        print(response)
        print("-" * 50)


# ── A：Student（无特权信息） ──────────────────────────────────────────────
student_msgs = build_student_messages(QUESTION)
generate(student_msgs, "A: Student prompt（无特权信息）")

# ── B：Teacher（仅 GT 数字答案） ─────────────────────────────────────────
teacher_b_msgs = build_teacher_privileged_messages(
    question=QUESTION,
    ground_truth=GROUND_TRUTH,
    reference_solution=None,
)
generate(teacher_b_msgs, "B: Teacher（仅告知答案=64，无推导过程）")

# ── C：Teacher（完整 reference solution） ───────────────────────────────
teacher_c_msgs = build_teacher_privileged_messages(
    question=QUESTION,
    ground_truth=GROUND_TRUTH,
    reference_solution=REFERENCE_SOLUTION,
)
generate(teacher_c_msgs, "C: Teacher（完整 reference solution）")

print("\n实验完成。")
