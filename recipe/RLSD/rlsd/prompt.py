"""
RLSD 的 prompt 模板。

包含：
  - Student prompt（仅问题）
  - Teacher privileged prompt（问题 + 参考解答或标答其一，用于 SD 分支的 ref forward）
  - Teacher context A：OPSD 风格（问题 + 正确答案）
  - Teacher context B：MRSD 风格（问题 + 错误轨迹 + 正确答案）
"""

# ──────────────────────────────────────────────────────────────
# 系统提示
# ──────────────────────────────────────────────────────────────

SYSTEM_STUDENT = "You are a helpful assistant."

SYSTEM_TEACHER = "You are a helpful assistant."

# ──────────────────────────────────────────────────────────────
# 模板函数
# ──────────────────────────────────────────────────────────────

MAX_WRONG_TRAJ_TOKENS = 1024  # §6.3：错误轨迹最多截断到此 token 数（字符数估算约 4x）
MAX_WRONG_TRAJ_CHARS = MAX_WRONG_TRAJ_TOKENS * 4  # 粗略估算

# 参考解答可能很长；特权 teacher prompt 在 tokenizer 前先做字符级截断，避免占满 max_prompt_len
MAX_REFERENCE_SOLUTION_CHARS = 32000


_USER_TAIL = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)
# 旧版 parquet / 历史脚本生成的后缀，还原题干时需一并剥离
_USER_TAIL_LEGACY = "\n\nNow provide a detailed step-by-step solution:"


def build_student_messages(question: str) -> list[dict]:
    """Student 的输入：仅包含问题，不含任何 hint。"""
    return [
        {"role": "system", "content": SYSTEM_STUDENT},
        {"role": "user", "content": f"Problem: {question}{_USER_TAIL}"},
    ]


def question_from_verl_prompt(prompt) -> str:
    """
    从 verl parquet 的 ``prompt`` 列（chat messages 列表）还原题干字符串。

    兼容：(a) 官方 GSM8K 等仅一条 user、正文即题干；(b) prepare_data / ``build_student_messages``
    存的 ``Problem: …`` + 当前或旧版 user 后缀。供 MRSDDataset 与 ``_evaluate`` 与训练侧共用，
    避免 eval 直接用 parquet 原始 messages 导致与 ``build_student_messages`` 不一致。

    注意：`pandas.read_parquet` 常把 ``prompt`` 列读成 ``numpy.ndarray``（元素为 dict），
    若不转换则会 ``str(prompt)`` 整列退化，评测题干错误。
    """
    # pandas → ndarray of dicts；Arrow/pyarrow 多为 list
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
            content = content[len(prefix) :]
        for tail in (_USER_TAIL, _USER_TAIL_LEGACY):
            if content.endswith(tail):
                content = content[: -len(tail)]
                break
        return content
    return str(prompt)


def build_teacher_privileged_messages(
    question: str,
    ground_truth: str,
    reference_solution: str | None = None,
) -> list[dict]:
    """
    SD 分支 Teacher：问题 + 可选参考推导；无参考推导时退化为一行标答 + 与旧版相同。

    有 ``reference_solution`` 时参考文中通常已含结论与 \\boxed{}，不再重复写标答行；
    无参考解答时仍注入 ``The correct final answer is: ...``，避免 teacher 失去特权信息。

    ``reference_solution`` 仅 teacher 可见；student 侧永远不可见。
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
    else:
        user_content = (
            f"Problem: {question}\n\n"
            f"The correct final answer is: {ground_truth}"
            f"{_USER_TAIL}"
        )
    return [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": user_content},
    ]


def _truncate_reference_solution(text: str, max_chars: int = MAX_REFERENCE_SOLUTION_CHARS) -> str:
    """截断数据集中的参考解答（与错误轨迹相同策略：保留头尾）。"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[reference solution truncated]...\n" + text[-half:]


def _truncate_wrong_traj(wrong_traj: str, max_chars: int = MAX_WRONG_TRAJ_CHARS) -> str:
    """
    截断过长的错误轨迹（§6.3）。
    策略：保留前 512 tokens + 后 512 tokens（字符估算）。
    """
    if len(wrong_traj) <= max_chars:
        return wrong_traj
    half = max_chars // 2
    return wrong_traj[:half] + "\n...[truncated]...\n" + wrong_traj[-half:]


def build_teacher_context_a(question: str, correct_answer: str) -> list[dict]:
    """
    Context A（OPSD 风格）：问题 + 正确答案。
    用于对照实验（§3.1 Context A）。
    """
    user_content = (
        f"Problem: {question}\n\n"
        f"I was told the correct answer is: {correct_answer}"
        f"{_USER_TAIL}"
    )
    return [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": user_content},
    ]


def build_teacher_context_b(
    question: str,
    wrong_traj: str,
    correct_answer: str,
) -> list[dict]:
    """
    Context B（MRSD 风格，§2.4）：问题 + 错误轨迹 + 正确答案。
    这是 MRSD 的核心 teacher context。
    """
    wrong_traj_truncated = _truncate_wrong_traj(wrong_traj)
    user_content = (
        f"Problem: {question}\n\n"
        f"My previous attempt (which was incorrect):\n{wrong_traj_truncated}\n\n"
        f"I was told the correct answer is: {correct_answer}"
        f"{_USER_TAIL}"
    )
    return [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": user_content},
    ]
