"""
实验：对比 Student prompt vs Teacher privileged prompt 下
Qwen2.5-3B-Instruct（原始权重，未训练）在问题 215 上的输出。

目的：验证 "reference solution" 捷径语言是否来自 teacher prompt 的直接诱导。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlsd.prompt import build_student_messages, build_teacher_privileged_messages
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "/data3/yyy/models/Qwen2.5-3B-Instruct"
N_SAMPLES = 3

QUESTION = (
    r"Let $(a_1,a_2,a_3,\ldots,a_{12})$ be a permutation of $(1,2,3,\ldots,12)$ for which  "
    r"$a_1 > a_2 > a_3 > a_4 > a_5 > a_6$ and $a_6 < a_7 < a_8 < a_9 < a_{10} < a_{11} < a_{12}.$ "
    r"An example of such a permutation is $(6,5,4,3,2,1,7,8,9,10,11,12).$ Find the number of such permutations."
)
GROUND_TRUTH = "462"

print("=" * 70)
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("模型加载完成\n")


def generate(messages, label, n=N_SAMPLES, max_new_tokens=1024):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    print(f"\n{'='*70}")
    print(f"【{label}】")
    print(f"Prompt 末尾（最后 300 字符）:\n...{text[-300:]}")
    print(f"\n--- 生成 {n} 条输出 ---")

    for i in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(i > 0),          # 第 1 条 greedy，其余 top-p
                temperature=0.7 if i > 0 else 1.0,
                top_p=0.9 if i > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        has_ref = "reference solution" in response.lower()
        has_correct = "462" in response
        tag = ""
        if has_ref:
            tag += " [⚠ reference solution]"
        if has_correct:
            tag += " [✓ 包含462]"
        else:
            tag += " [✗ 未出现462]"

        print(f"\n[样本 {i+1}]{'greedy' if i==0 else 'top-p'}{tag}")
        print(response)
        print("-" * 40)


# ── A：Student prompt（无特权信息） ──────────────────────────────────────
student_msgs = build_student_messages(QUESTION)
generate(student_msgs, "A: Student prompt（无特权信息）")

# ── B：Teacher privileged prompt（只有 GT 答案，无 reference_solution） ──
teacher_msgs = build_teacher_privileged_messages(
    question=QUESTION,
    ground_truth=GROUND_TRUTH,
    reference_solution=None,   # 训练数据中此字段为空
)
generate(teacher_msgs, "B: Teacher privileged prompt（含 GT=462，无参考解答）")

print("\n实验完成。")
