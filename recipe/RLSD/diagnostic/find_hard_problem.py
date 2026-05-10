"""找一道模型答错、但有完整 solution 的题，用于内化实验"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rlsd.prompt import build_student_messages

MODEL_PATH = "/data3/yyy/models/Qwen2.5-3B-Instruct"
PARQUET_PATH = "/data3/yyy/verl/data/Openthoughts_math_30k_opsd/data/train.parquet"

df = pd.read_parquet(PARQUET_PATH)
candidates = df[
    (df['solution'].str.len() > 600) &
    (df['solution'].str.len() < 2000) &
    (df['correct'] == True) &
    (df['Answer'].str.len() < 20)
].head(80)
print(f"候选题数: {len(candidates)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()
print("模型加载完成，开始筛题...\n")

hard = []
for idx, row in candidates.iterrows():
    msgs = build_student_messages(row["problem"])
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=700, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    ans = str(row["Answer"]).strip()
    correct = ans in resp
    status = "[ok]   " if correct else "[WRONG]"
    print(f"{status} idx={idx:4d} | ans={ans:15s} | Q: {row['problem'][:80]}")
    if not correct:
        hard.append(idx)
        print(f"         Model tail: ...{resp[-120:]}\n")
    if len(hard) >= 5:
        print("\n找到足够难题，停止。")
        break

print(f"\n答错的题 idx 列表: {hard}")
