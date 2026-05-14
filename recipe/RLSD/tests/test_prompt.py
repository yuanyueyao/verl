"""
测试 prompt.py 中 student / teacher 模板的输出是否符合预期。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from recipe.RLSD.rlsd.prompt import (
    SYSTEM_STUDENT,
    SYSTEM_TEACHER,
    MAX_REFERENCE_SOLUTION_CHARS,
    build_student_messages,
    build_teacher_privileged_messages,
    question_from_verl_prompt,
    _truncate_reference_solution,
)

_STUDENT_USER_TAIL = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# ═══════════════════════════════════════════════════════════
# 测试数据
# ═══════════════════════════════════════════════════════════

SAMPLE_QUESTION = (
    "Let $f(x) = x^3 - 3x + 1$. Find all real roots of $f(x) = 0$."
)
SAMPLE_ANSWER = "x = 2\\cos(20°),\\; 2\\cos(140°),\\; 2\\cos(260°)"
SAMPLE_SOLUTION = (
    "Let x = 2 cos t. Then x^3 - 3x + 1 = 8 cos^3 t - 6 cos t + 1 = 2(4 cos^3 t - 3 cos t) + 1 = 2 cos(3t) + 1 = 0. "
    "So cos(3t) = -1/2, giving 3t = 120°, 240°, 480°, ... "
    "Thus x = 2 cos(40°), 2 cos(80°), 2 cos(160°)."
)

SEPARATOR = "─" * 60


def pprint_messages(messages: list[dict], label: str):
    """打印 chat messages。"""
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[{role}]")
        print(content)
    print()


# ═══════════════════════════════════════════════════════════
# Test 1: build_student_messages
# ═══════════════════════════════════════════════════════════

def test_student_messages():
    print("=" * 60)
    print("TEST 1: build_student_messages")
    print("=" * 60)

    msgs = build_student_messages(SAMPLE_QUESTION)
    pprint_messages(msgs, "Student Prompt")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[0]["content"] == SYSTEM_STUDENT
    assert "\\boxed{}" in msgs[1]["content"]
    assert msgs[1]["content"] == f"Problem: {SAMPLE_QUESTION}{_STUDENT_USER_TAIL}"
    assert SAMPLE_ANSWER not in msgs[1]["content"]

    print("✓ 结构正确：2 条消息 (system + user)")
    print("✓ user 末尾包含 \\boxed{}")
    print("✓ user 内容仅包含问题，无答案泄漏")


# ═══════════════════════════════════════════════════════════
# Test 2: build_teacher_privileged_messages（solution-conditioned）
# ═══════════════════════════════════════════════════════════

def test_teacher_solution_conditioned():
    print("\n" + "=" * 60)
    print("TEST 2: build_teacher_privileged_messages（solution-conditioned）")
    print("=" * 60)

    msgs = build_teacher_privileged_messages(SAMPLE_QUESTION, SAMPLE_ANSWER, SAMPLE_SOLUTION)
    pprint_messages(msgs, "Teacher（含完整 solution）")

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[0]["content"] == SYSTEM_TEACHER

    user_content = msgs[1]["content"]
    assert f"Problem: {SAMPLE_QUESTION}" in user_content
    assert "reference solution" in user_content.lower()
    assert SAMPLE_SOLUTION in user_content
    assert "reason step by step" in user_content.lower()

    print("✓ 结构正确")
    print("✓ 包含完整 reference solution")
    print("✓ 不含仅答案的捷径语言")


def test_teacher_no_solution():
    """无 reference_solution 时不应注入 answer-only 内容。"""
    msgs = build_teacher_privileged_messages(SAMPLE_QUESTION, SAMPLE_ANSWER, reference_solution=None)
    user_content = msgs[1]["content"]

    # 不应包含 "correct final answer is" 这种 answer-only 引导
    assert "correct final answer is" not in user_content.lower()
    assert SAMPLE_ANSWER not in user_content
    print("✓ 无 reference_solution 时不注入答案")


# ═══════════════════════════════════════════════════════════
# Test 3: _truncate_reference_solution
# ═══════════════════════════════════════════════════════════

def test_truncation():
    print("\n" + "=" * 60)
    print("TEST 3: _truncate_reference_solution")
    print("=" * 60)

    short = "Short solution."
    result = _truncate_reference_solution(short)
    assert result == short
    print(f"✓ 短文本 ({len(short)} chars) 未截断")

    exact = "x" * MAX_REFERENCE_SOLUTION_CHARS
    result = _truncate_reference_solution(exact)
    assert result == exact
    print(f"✓ 边界文本 ({len(exact)} chars) 未截断")

    long_text = "A" * 1000 + "MIDDLE" + "Z" * (MAX_REFERENCE_SOLUTION_CHARS + 1000)
    result = _truncate_reference_solution(long_text)
    assert len(result) < len(long_text)
    assert "...[reference solution truncated]..." in result
    assert result.startswith("A")
    assert result.endswith("Z")
    print(f"✓ 超长文本 ({len(long_text)} chars) 被截断到 ~{len(result)} chars")


# ═══════════════════════════════════════════════════════════
# Test 4: tokenizer 集成
# ═══════════════════════════════════════════════════════════

def test_with_tokenizer():
    print("\n" + "=" * 60)
    print("TEST 4: tokenizer chat_template 集成")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("⚠ transformers 未安装，跳过")
        return

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"加载 tokenizer: {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for label, msgs_fn in [
        ("Student", lambda: build_student_messages(SAMPLE_QUESTION)),
        ("Teacher+Solution", lambda: build_teacher_privileged_messages(
            SAMPLE_QUESTION, SAMPLE_ANSWER, SAMPLE_SOLUTION)),
    ]:
        msgs = msgs_fn()
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        token_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

        print(f"\n--- {label} ---")
        print(f"  消息数: {len(msgs)}")
        print(f"  渲染后字符数: {len(text)}")
        print(f"  Token 数: {len(token_ids)}")
        assert len(text) > 0
        assert len(token_ids) > 0
        decoded = tok.decode(token_ids, skip_special_tokens=False)
        assert len(decoded) > 0

    print("\n✓ 所有 prompt 模板与 tokenizer chat_template 兼容")


# ═══════════════════════════════════════════════════════════
# Test 5: question_from_verl_prompt
# ═══════════════════════════════════════════════════════════

def test_question_from_verl_prompt_roundtrip():
    msgs = build_student_messages(SAMPLE_QUESTION)
    assert question_from_verl_prompt(msgs) == SAMPLE_QUESTION


def test_question_from_verl_prompt_gsm8k_style():
    raw = "Janet's ducks lay 16 eggs per day."
    prompt = [{"role": "user", "content": raw}]
    assert question_from_verl_prompt(prompt) == raw


def test_question_from_verl_prompt_problem_prefix():
    raw = "Compute 1+1."
    prompt = [
        {"role": "system", "content": SYSTEM_STUDENT},
        {"role": "user", "content": f"Problem: {raw}"},
    ]
    assert question_from_verl_prompt(prompt) == raw


# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_student_messages()
    test_teacher_solution_conditioned()
    test_teacher_no_solution()
    test_truncation()
    test_with_tokenizer()
    test_question_from_verl_prompt_roundtrip()
    test_question_from_verl_prompt_gsm8k_style()
    test_question_from_verl_prompt_problem_prefix()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
