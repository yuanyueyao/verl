"""
RLSD 的 prompt 模板。

包含：
  - Student prompt（仅问题）
  - Teacher privileged prompt（问题 + 完整参考解答，用于 SD 分支的 ref forward）
"""

# ──────────────────────────────────────────────────────────────
# 系统提示
# ──────────────────────────────────────────────────────────────

SYSTEM_STUDENT = "You are a helpful assistant."

SYSTEM_TEACHER = "You are a helpful assistant."

# ──────────────────────────────────────────────────────────────
# 模板函数
# ──────────────────────────────────────────────────────────────

# 参考解答可能很长；特权 teacher prompt 在 tokenizer 前先做字符级截断，避免占满 max_prompt_len
MAX_REFERENCE_SOLUTION_CHARS = 32000


_USER_TAIL = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def build_student_messages(question: str) -> list[dict]:
    """Student 的输入：仅包含问题，不含任何 hint。"""
    return [
        {"role": "system", "content": SYSTEM_STUDENT},
        {"role": "user", "content": f"Problem: {question}{_USER_TAIL}"},
    ]


def question_from_verl_prompt(prompt) -> str:
    """
    从 verl parquet 的 ``prompt`` 列（chat messages 列表）还原题干字符串。

    注意：`pandas.read_parquet` 常把 ``prompt`` 列读成 ``numpy.ndarray``（元素为 dict），
    若不转换则会 ``str(prompt)`` 整列退化，评测题干错误。
    """
    if hasattr(prompt, "tolist") and not isinstance(prompt, (list, dict, str)):
        try:
            prompt = prompt.tolist()
        except Exception:
            pass
    elif isinstance(prompt, tuple):
        prompt = list(prompt)

    if isinstance(prompt, list) and len(prompt) > 0:
        last = prompt[-1]
        if not isinstance(last, dict):
            return str(last)
        content = last.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        prefix = "Problem: "
        if content.startswith(prefix):
            content = content[len(prefix):]
        return content
    return str(prompt)


def build_teacher_privileged_messages(
    question: str,
    ground_truth: str,
    reference_solution: str | None = None,
) -> list[dict]:
    """
    SD 分支 Teacher：问题 + 完整参考解答。

    Teacher 看到 reference_solution（含完整推导过程），能内化并产出自洽的正确推理。
    无 reference_solution 时退化为仅给出 GT 答案——此前实验已证明这会导致捷径学习，不应该使用。
    """
    ref = (reference_solution or "").strip()
    ref_block = ""
    if ref:
        ref_block = (
            "\n\nBelow is a verified reference solution showing how the answer is derived. "
            "Use it to reason about the problem; your own response wording may differ.\n\n"
            f"{_truncate_reference_solution(ref)}\n"
        )
    user_content = f"Problem: {question}{ref_block}{_USER_TAIL}"
    return [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": user_content},
    ]


def _truncate_reference_solution(text: str, max_chars: int = MAX_REFERENCE_SOLUTION_CHARS) -> str:
    """截断过长的参考解答（保留头尾）。"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[reference solution truncated]...\n" + text[-half:]
