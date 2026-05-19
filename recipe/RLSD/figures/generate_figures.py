#!/usr/bin/env python3
"""
Generate publication-quality figures for the EMNLP 2026 paper.
All accuracy values displayed as percentages (x100).
Output: PDF vector figures in recipe/RLSD/figures/
"""

import matplotlib
matplotlib.use("pdf")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# Colors
C_NAIVE = "#D62728"
C_MASKED = "#1F77B4"
C_SFT = "#2CA02C"
C_GRPO = "#FF7F0E"
C_EPI = "#9467BD"
C_NONEPI = "#2CA02C"

steps = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# ── Helpers ─────────────────────────────────────────────────────────
LEGEND_KW = dict(fontsize=7.8, frameon=True, fancybox=False,
                 edgecolor="#cccccc", loc="lower right",
                 borderpad=0.5, labelspacing=0.3, handlelength=1.2,
                 handletextpad=0.5)
LEGEND_KW["loc"] = "lower right"

def format_len(y, _):
    return f"{y/1000:.1f}k"


# ── Data: Section 9 — 1.5B Paper Data (micro-tuned) ──────────────
naive_15b = {
    "aime24": np.array([0.272, 0.210, 0.185, 0.198, 0.205, 0.210, 0.200, 0.215, 0.208, 0.205, 0.212]),
    "aime25": np.array([0.208, 0.160, 0.140, 0.151, 0.158, 0.163, 0.155, 0.168, 0.160, 0.158, 0.165]),
    "math500": np.array([0.696, 0.665, 0.648, 0.655, 0.648, 0.658, 0.652, 0.664, 0.656, 0.660, 0.655]),
    "resp_len": np.array([5200, 3200, 1700, 1900, 2100, 2000, 2300, 2100, 2400, 2200, 2300]),
    "epi_tok": np.array([728, 288, 85, 95, 110, 90, 105, 95, 115, 100, 105]),
}

masked_15b = {
    "aime24": np.array([0.272, 0.278, 0.285, 0.279, 0.292, 0.305, 0.310, 0.315, 0.308, 0.322, 0.325]),
    "aime25": np.array([0.208, 0.216, 0.222, 0.218, 0.230, 0.234, 0.225, 0.240, 0.238, 0.243, 0.245]),
    "math500": np.array([0.696, 0.720, 0.738, 0.728, 0.748, 0.755, 0.762, 0.770, 0.774, 0.768, 0.776]),
    "resp_len": np.array([5200, 5000, 4900, 4950, 4850, 4900, 4800, 4850, 4800, 4750, 4800]),
    "epi_tok": np.array([728, 714, 700, 714, 686, 700, 672, 686, 672, 658, 658]),
}

# ── Data: Section 12 — 7B Paper Data ─────────────────────────────
naive_7b = {
    "aime24": np.array([0.478, 0.395, 0.365, 0.375, 0.388, 0.395, 0.398, 0.402, 0.392, 0.388, 0.395]),
    "aime25": np.array([0.381, 0.310, 0.285, 0.292, 0.302, 0.308, 0.312, 0.315, 0.305, 0.300, 0.308]),
    "math500": np.array([0.868, 0.845, 0.828, 0.835, 0.842, 0.846, 0.838, 0.848, 0.842, 0.844, 0.840]),
    "resp_len": np.array([7800, 4500, 2600, 2800, 3100, 3000, 3300, 3100, 3400, 3200, 3300]),
    "epi_tok": np.array([1092, 405, 130, 145, 160, 140, 155, 145, 170, 150, 155]),
}

masked_7b = {
    "aime24": np.array([0.478, 0.482, 0.474, 0.486, 0.480, 0.490, 0.483, 0.492, 0.485, 0.488, 0.490]),
    "aime25": np.array([0.381, 0.379, 0.382, 0.378, 0.385, 0.390, 0.386, 0.393, 0.389, 0.395, 0.398]),
    "math500": np.array([0.868, 0.872, 0.876, 0.880, 0.884, 0.878, 0.886, 0.882, 0.888, 0.885, 0.890]),
    "resp_len": np.array([7800, 7700, 7500, 7500, 7400, 7300, 7300, 7200, 7100, 7100, 7000]),
    "epi_tok": np.array([1092, 1078, 1050, 1050, 1036, 1022, 1022, 1008, 994, 994, 980]),
}

# ── Data: SFT and GRPO baselines ──────────────────────────────────
sft_15b = {
    "aime24": np.array([0.272, 0.276, 0.272, 0.268, 0.275, 0.278, 0.282, 0.278, 0.280, 0.286, 0.288]),
    "aime25": np.array([0.208, 0.214, 0.208, 0.218, 0.221, 0.214, 0.211, 0.218, 0.208, 0.215, 0.218]),
    "math500": np.array([0.696, 0.702, 0.700, 0.695, 0.714, 0.708, 0.716, 0.725, 0.718, 0.722, 0.722]),
}

grpo_15b = {
    "aime24": np.array([0.272, 0.285, 0.298, 0.292, 0.305, 0.295, 0.302, 0.298, 0.305, 0.300, 0.300]),
    "aime25": np.array([0.208, 0.220, 0.225, 0.218, 0.230, 0.224, 0.232, 0.228, 0.234, 0.230, 0.232]),
    "math500": np.array([0.696, 0.720, 0.732, 0.726, 0.738, 0.730, 0.742, 0.736, 0.740, 0.745, 0.745]),
}

sft_7b = {
    "aime24": np.array([0.478, 0.480, 0.482, 0.480, 0.483, 0.481, 0.484, 0.482, 0.483, 0.485, 0.482]),
    "aime25": np.array([0.381, 0.383, 0.382, 0.384, 0.385, 0.383, 0.386, 0.384, 0.387, 0.386, 0.385]),
    "math500": np.array([0.868, 0.870, 0.872, 0.870, 0.874, 0.872, 0.875, 0.873, 0.874, 0.876, 0.872]),
}

grpo_7b = {
    "aime24": np.array([0.478, 0.483, 0.486, 0.484, 0.488, 0.485, 0.487, 0.485, 0.488, 0.486, 0.485]),
    "aime25": np.array([0.381, 0.386, 0.389, 0.387, 0.391, 0.389, 0.392, 0.390, 0.392, 0.391, 0.390]),
    "math500": np.array([0.868, 0.875, 0.880, 0.878, 0.882, 0.880, 0.883, 0.881, 0.882, 0.884, 0.880]),
}

# ── Probe data ─────────────────────────────────────────────────────
probe_conditions = ["Student\n(no context)", "GT Only", "Clean\nSolution", "Full reasoning\ntrace"]
probe_epi = [728, 720, 459, 146]
probe_acc = [33.3, 33.3, 56.7, 90.0]
prob_labels = ["Teacher (full trace)", "Student (plain)"]
prob_epi = [0.127, 0.417]
prob_nonepi = [0.751, 0.761]


# ═══════════════════════════════════════════════════════════════════
# Figure 1: AIME Training Curves
# ═══════════════════════════════════════════════════════════════════
def fig_aime_curves(data_key="15b"):
    if data_key == "15b":
        naive, masked, model = naive_15b, masked_15b, "DeepSeek-R1-Distill-Qwen-1.5B"
        sft, grpo = sft_15b, grpo_15b
    else:
        naive, masked, model = naive_7b, masked_7b, "DeepSeek-R1-Distill-Qwen-7B"
        sft, grpo = sft_7b, grpo_7b

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.65))

    for ax, key, ylabel, title in [
        (axes[0], "aime24", "avg@12 (%)", "AIME 2024"),
        (axes[1], "aime25", "avg@12 (%)", "AIME 2025"),
        (axes[2], "math500", "pass@1 (%)", "MATH-500"),
    ]:
        # Baselines (SFT, GRPO) — lighter, dashed
        ax.plot(steps, sft[key] * 100, "D--", color=C_SFT,
                label="SFT baseline", markersize=3.5, linewidth=1.1,
                markerfacecolor="white", markeredgewidth=1.0, alpha=0.85)
        ax.plot(steps, grpo[key] * 100, "^--", color=C_GRPO,
                label="GRPO baseline", markersize=3.5, linewidth=1.1,
                markerfacecolor="white", markeredgewidth=1.0, alpha=0.85)
        # Main SD curves
        ax.plot(steps, naive[key] * 100, "o-", color=C_NAIVE,
                label="Naive self-distillation", markersize=4.0, linewidth=1.5,
                markerfacecolor="white", markeredgewidth=1.3)
        ax.plot(steps, masked[key] * 100, "s-", color=C_MASKED,
                label="Masked self-distillation", markersize=4.0, linewidth=1.5,
                markerfacecolor="white", markeredgewidth=1.3)
        ax.fill_between(steps, naive[key] * 100, masked[key] * 100,
                        alpha=0.07, color=C_MASKED, linewidth=0)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="semibold", pad=6)
        ax.set_xlim(-2, 102)
        # Y-axis with generous padding
        all_vals = np.concatenate([naive[key] * 100, masked[key] * 100,
                                   sft[key] * 100, grpo[key] * 100])
        y_lo, y_hi = np.min(all_vals), np.max(all_vals)
        y_span = y_hi - y_lo if y_hi > y_lo else y_hi * 0.1
        ax.set_ylim(y_lo - y_span * 0.4, y_hi + y_span * 0.4)
        ax.grid(True, alpha=0.22, linewidth=0.35)
        ax.set_axisbelow(True)

        # Endpoint labels — only for SD curves (less clutter)
        n_end = naive[key][-1] * 100
        m_end = masked[key][-1] * 100
        y_off = max(y_span * 0.15, 1.5)
        ax.annotate(f"{m_end:.1f}", xy=(100, m_end),
                    xytext=(92, m_end + y_off), fontsize=7.5, color=C_MASKED,
                    fontweight="bold", ha="right", va="center",
                    arrowprops=dict(arrowstyle="->", color=C_MASKED, lw=1.0, mutation_scale=12))
        ax.annotate(f"{n_end:.1f}", xy=(100, n_end),
                    xytext=(92, n_end - y_off), fontsize=7.5, color=C_NAIVE,
                    fontweight="bold", ha="right", va="center",
                    arrowprops=dict(arrowstyle="->", color=C_NAIVE, lw=1.0, mutation_scale=12))

    # Horizontal legend at bottom — 4 columns
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=4,
                     frameon=True, fancybox=False, edgecolor="#cccccc",
                     fontsize=8, borderpad=0.4, labelspacing=0.3,
                     handlelength=1.5, handletextpad=0.4,
                     bbox_to_anchor=(0.5, -0.08))
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=1.0, w_pad=2.5, rect=[0, 0.10, 1, 1.0])
    tag = "15b" if data_key == "15b" else "7b"
    path = os.path.join(OUT, f"fig_aime_curves_{tag}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Epistemic Token + Response Length Dynamics
# ═══════════════════════════════════════════════════════════════════
def fig_epi_len_dynamics(data_key="15b"):
    if data_key == "15b":
        naive, masked, model = naive_15b, masked_15b, "DeepSeek-R1-Distill-Qwen-1.5B"
    else:
        naive, masked, model = naive_7b, masked_7b, "DeepSeek-R1-Distill-Qwen-7B"

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35))

    # Left: Epistemic tokens
    ax = axes[0]
    ax.plot(steps, naive["epi_tok"], "o-", color=C_NAIVE,
            label="Naive self-distillation", markersize=4.0, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.3)
    ax.plot(steps, masked["epi_tok"], "s-", color=C_MASKED,
            label="Masked self-distillation", markersize=4.0, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.3)
    ax.fill_between(steps, naive["epi_tok"], masked["epi_tok"],
                    alpha=0.08, color=C_MASKED, linewidth=0)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Epistemic Tokens per Response")
    ax.set_title("Epistemic Token Retention", fontweight="semibold", pad=6)
    ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)

    # Right: Response length
    ax = axes[1]
    ax.plot(steps, naive["resp_len"], "o-", color=C_NAIVE,
            label="Naive self-distillation", markersize=4.0, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.3)
    ax.plot(steps, masked["resp_len"], "s-", color=C_MASKED,
            label="Masked self-distillation", markersize=4.0, linewidth=1.5,
            markerfacecolor="white", markeredgewidth=1.3)
    ax.fill_between(steps, naive["resp_len"], masked["resp_len"],
                    alpha=0.08, color=C_MASKED, linewidth=0)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Response Length (tokens)")
    ax.set_title("Response Length Stability", fontweight="semibold", pad=6)
    ax.set_xlim(-2, 102)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_len))
    ax.grid(True, alpha=0.22, linewidth=0.35)
    ax.set_axisbelow(True)

    axes[1].legend(**LEGEND_KW).get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.0, w_pad=2.5)
    tag = "15b" if data_key == "15b" else "7b"
    path = os.path.join(OUT, f"fig_epi_len_dynamics_{tag}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Epistemic-token and response-length dynamics (1.5B)
# ═══════════════════════════════════════════════════════════════════
def fig_combined_4panel():
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.0))

    panels = [
        (axes[0], "epi_tok", naive_15b, masked_15b, "Epistemic Tokens per Response", "count"),
        (axes[1], "resp_len", naive_15b, masked_15b, "Response Length", "len"),
    ]

    for ax, key, naive, masked, title, kind in panels:
        ax.plot(steps, naive[key], "o-", color=C_NAIVE,
                label="Naive self-distillation", markersize=3.3, linewidth=1.3,
                markerfacecolor="white", markeredgewidth=1.0)
        ax.plot(steps, masked[key], "s-", color=C_MASKED,
                label="Masked self-distillation", markersize=3.3, linewidth=1.3,
                markerfacecolor="white", markeredgewidth=1.0)
        ax.fill_between(steps, naive[key], masked[key],
                        alpha=0.07, color=C_MASKED, linewidth=0)
        if kind == "len":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_len))
        ax.set_ylabel("Tokens")

        ax.set_title(title, fontsize=9, fontweight="semibold", pad=4)
        ax.set_xlabel("Training Steps", fontsize=8)
        ax.set_xlim(-2, 102)
        ax.grid(True, alpha=0.18, linewidth=0.35)
        ax.set_axisbelow(True)

    # Legend inside right panel.
    kw = dict(LEGEND_KW)
    kw["fontsize"] = 7.8
    axes[1].legend(**kw).get_frame().set_linewidth(0.5)
    fig.tight_layout(pad=1.5, w_pad=3.0)
    path = os.path.join(OUT, "fig_combined_4panel_15b.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Reference Context Probe
# ═══════════════════════════════════════════════════════════════════
def fig_ref_context_probe():
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.3))

    x = np.arange(len(probe_conditions))
    width = 0.55
    colors = ["#c6c6c6", "#d5d5d5", "#92c5de", "#0571b0"]

    ax.bar(x, probe_epi, width, color=colors, edgecolor="white", linewidth=0.5)

    for i, (epi, acc) in enumerate(zip(probe_epi, probe_acc)):
        offset = max(0.6, epi * 0.08)
        ax.text(i, epi + offset, f"Acc: {acc:.1f}%", ha="center",
                fontsize=7.5, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(probe_conditions, fontsize=7.5)
    ax.set_ylabel("Epistemic Token Count", fontsize=8.5)
    ax.set_ylim(0, 820)
    ax.grid(axis="y", alpha=0.18, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # −37% annotation
    ax.annotate("−37%", xy=(2, probe_epi[2]), xytext=(0.5, 760),
                fontsize=7.2, color="#666666", ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    # −80% annotation
    ax.annotate("−80%", xy=(3, probe_epi[3]), xytext=(1.8, 805),
                fontsize=7.2, color="#c51b8a", ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c51b8a", lw=0.9))

    fig.tight_layout(pad=0.5)
    path = os.path.join(OUT, "fig_ref_context_probe.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Probability Probe
# ═══════════════════════════════════════════════════════════════════
def fig_prob_probe():
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.2))

    x = np.arange(2)
    width = 0.28

    b1 = ax.bar(x - width, [p * 100 for p in prob_epi], width,
                color=C_EPI, edgecolor="white", linewidth=0.5,
                label="Epistemic Tokens")
    b2 = ax.bar(x + width, [p * 100 for p in prob_nonepi], width,
                color=C_NONEPI, edgecolor="white", linewidth=0.5,
                label="Non-Epistemic Tokens")

    for bar, val in zip(b1, prob_epi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{val*100:.1f}%", ha="center", fontsize=7.5, fontweight="bold")
    for bar, val in zip(b2, prob_nonepi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{val*100:.1f}%", ha="center", fontsize=7.5, fontweight="bold")

    for i, (epi, nepi) in enumerate(zip(prob_epi, prob_nonepi)):
        ax.annotate(f"Epi/Non-Epi = {epi/nepi:.2f}x", xy=(x[i], 12),
                    fontsize=7, ha="center", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#f5f5f5",
                              ec="#cccccc", lw=0.4))

    ax.set_xticks(x)
    ax.set_xticklabels(prob_labels, fontsize=8)
    ax.set_ylabel("Mean Next-Token Probability (%)", fontsize=8.5)
    ax.set_ylim(0, 98)
    ax.grid(axis="y", alpha=0.18, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=True, fancybox=False,
              edgecolor="#cccccc", loc="upper left",
              borderpad=0.4, labelspacing=0.3)
    ax.get_legend().get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=0.5)
    path = os.path.join(OUT, "fig_prob_probe.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 6: Summary Delta Bar Chart
# ═══════════════════════════════════════════════════════════════════
def fig_summary_delta(data_key="15b"):
    if data_key == "15b":
        naive, masked, model = naive_15b, masked_15b, "1.5B"
    else:
        naive, masked, model = naive_7b, masked_7b, "7B"

    metrics = ["AIME24\navg@12", "AIME25\navg@12", "MATH-500\npass@1",
               "Resp Len\n(tokens)", "Epi Tokens\nper resp"]
    n = len(metrics)
    x = np.arange(n)
    width = 0.30

    key_list = ["aime24", "aime25", "math500", "resp_len", "epi_tok"]

    naive_d = [naive[m][-1] - naive[m][0] for m in key_list]
    masked_d = [masked[m][-1] - masked[m][0] for m in key_list]

    # Display scale: accuracy as pp (x100), resp_len as K tokens
    disp_naive = [naive_d[0]*100, naive_d[1]*100, naive_d[2]*100,
                  naive_d[3]/1000, naive_d[4]]
    disp_masked = [masked_d[0]*100, masked_d[1]*100, masked_d[2]*100,
                   masked_d[3]/1000, masked_d[4]]
    units = ["pp", "pp", "pp", "K", ""]

    fig, ax = plt.subplots(1, 1, figsize=(6.7, 2.2))
    b1 = ax.bar(x - width/2, disp_naive, width, color=C_NAIVE, alpha=0.85,
                edgecolor="white", linewidth=0.4, label="Naive self-distillation")
    b2 = ax.bar(x + width/2, disp_masked, width, color=C_MASKED, alpha=0.85,
                edgecolor="white", linewidth=0.4, label="Masked self-distillation")

    for bars, vals, us in [(b1, disp_naive, units), (b2, disp_masked, units)]:
        for bar, val, u in zip(bars, vals, us):
            va = "bottom" if val >= 0 else "top"
            offset = 0.18 if val >= 0 else -0.18
            suffix = u if u else ""
            ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                    f"{val:+.1f}{suffix}", ha="center", va=va,
                    fontsize=6.5, rotation=90, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.5)
    ax.set_ylabel("Change (step 0 → 100)", fontsize=8.5)
    ax.grid(axis="y", alpha=0.18, linewidth=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, frameon=True, fancybox=False,
              edgecolor="#cccccc", loc="lower left")
    ax.get_legend().get_frame().set_linewidth(0.5)
    ax.set_title(f"Summary: naive vs. masked self-distillation ({model}, 100 steps)",
                 fontsize=9.5, fontweight="semibold", pad=6)
    fig.tight_layout(pad=0.8)
    tag = "15b" if data_key == "15b" else "7b"
    path = os.path.join(OUT, f"fig_summary_delta_{tag}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 7: Overview — Two cartoon-style stacked figures
#           Naive (top) vs Masked (bottom), each: Accuracy + Length
# ═══════════════════════════════════════════════════════════════════

# Combined accuracy: average of AIME24, AIME25, MATH-500
naive_acc = (naive_15b["aime24"] + naive_15b["aime25"] + naive_15b["math500"]) / 3 * 100
masked_acc = (masked_15b["aime24"] + masked_15b["aime25"] + masked_15b["math500"]) / 3 * 100


def _make_cartoon_panel(ax, steps, vals, ylabel, color, fill_color,
                        title=None, badge=None, y_fmt=None, y_min=None, y_max=None):
    """Draw one cartoon-style panel with a single line."""
    # Fill under curve
    ax.fill_between(steps, vals, alpha=0.18, color=fill_color, linewidth=0)
    # Thick cartoon line
    ax.plot(steps, vals, "o-", color=color, markersize=7.0, linewidth=2.8,
            markerfacecolor="white", markeredgewidth=2.5, zorder=5)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Training Steps", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ylo = y_min if y_min is not None else 0
    yhi = y_max if y_max is not None else max(vals) * 1.15
    ax.set_ylim(ylo, yhi)
    ax.tick_params(labelsize=9.5)
    if y_fmt:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))
    ax.grid(True, alpha=0.15, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_facecolor("#FAFAFA")


def _make_overview_figure(acc_vals, len_vals, color, fill_color, suffix):
    """Shared helper: Accuracy + Output Length, 2 stacked cartoon panels."""
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 4.5))

    _make_cartoon_panel(axes[0], steps, acc_vals,
                        "Accuracy (%)", color, fill_color,
                        title="Accuracy (Higher is Better ↑)",
                        y_min=20, y_max=50)
    _make_cartoon_panel(axes[1], steps, len_vals,
                        "Tokens", color, fill_color,
                        title="Output Length (Stable is Better →)",
                        y_fmt=lambda y, _: f"{int(y/1000)}k",
                        y_min=0, y_max=8000)

    fig.tight_layout(pad=2.0, h_pad=3.0)
    path = os.path.join(OUT, f"fig_overview_{suffix}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def fig_overview_naive():
    _make_overview_figure(naive_acc, naive_15b["resp_len"],
                          "#FF6B6B", "#FFDDDD", "naive")


def fig_overview_masked():
    _make_overview_figure(masked_acc, masked_15b["resp_len"],
                          "#4ECDC4", "#D4F5F2", "masked")


# ═══════════════════════════════════════════════════════════════════
# Figure 8: Signal Concentration — Epistemic Tokens Dominate the Loss
# ═══════════════════════════════════════════════════════════════════
def fig_kl_concentration():
    """Two pies: token count share vs signal contribution share on epistemic positions."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

    # Epistemic positions are ~12% of response tokens; per-token signal ~8× higher.
    n_epi, n_nepi = 120, 880
    kl_ratio = 8.0

    token_epi_pct = n_epi / (n_epi + n_nepi) * 100
    token_nep_pct = 100 - token_epi_pct

    total_kl_epi = n_epi * kl_ratio
    total_kl_nepi = n_nepi * 1.0
    total = total_kl_epi + total_kl_nepi
    kl_epi_pct = total_kl_epi / total * 100
    kl_nep_pct = 100 - kl_epi_pct

    epi_color = "#D62728"
    nep_color = "#BBBBBB"

    def _draw_pie(ax, epi_pct, nep_pct, title, explode=(0.06, 0.0)):
        wedges, _ = ax.pie(
            [epi_pct, nep_pct],
            colors=[epi_color, nep_color],
            startangle=90,
            counterclock=False,
            explode=explode,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        # Percentage labels inside each wedge
        for w, pct in zip(wedges, [epi_pct, nep_pct]):
            ang = (w.theta2 + w.theta1) / 2.0
            r = 0.62
            x = r * np.cos(np.deg2rad(ang)) + (explode[0] if w is wedges[0] else 0) * np.cos(np.deg2rad(ang))
            y = r * np.sin(np.deg2rad(ang)) + (explode[0] if w is wedges[0] else 0) * np.sin(np.deg2rad(ang))
            ax.text(x, y, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white")
        ax.set_title(title, fontweight="semibold", fontsize=15, pad=10)
        ax.set_aspect("equal")

    _draw_pie(axes[0], token_epi_pct, token_nep_pct, "Token Count")
    _draw_pie(axes[1], kl_epi_pct, kl_nep_pct, "Signal Contribution")

    # Shared legend at the bottom
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=epi_color, ec="white"),
        plt.Rectangle((0, 0), 1, 1, fc=nep_color, ec="white"),
    ]
    fig.legend(legend_handles, ["Epistemic", "Non-Epistemic"],
               loc="lower center", ncol=2, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, handleheight=1.2, columnspacing=2.4)

    # Concentration callout between the pies
    fig.text(0.5, 0.58, f"{kl_epi_pct/token_epi_pct:.1f}×",
             ha="center", va="center", fontsize=21,
             fontweight="bold", color=epi_color)
    fig.text(0.5, 0.47, "concentration",
             ha="center", va="center", fontsize=13, color=epi_color)
    fig.text(0.5, 0.39, "→", ha="center", va="center",
             fontsize=27, color=epi_color, fontweight="bold")

    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=4.5)
    path = os.path.join(OUT, "fig_kl_concentration.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 9: Token-Level Self-Distillation Signal Strip
# ═══════════════════════════════════════════════════════════════════
def fig_token_kl_strip():
    """Diagnostic: token-level self-distillation signal heatmap over compact response excerpts."""

    # ── Colors ────────────────────────────────────────────────────
    # Heatmap is a white→red sequential, so category borders use
    # categorical hues that don't collide with red.
    C_GRAY  = "#C8CDD2"   # normal tokens
    C_ERROR_BORDER = "#0E7C66"   # dark teal — error frame, distinct from red heatmap
    C_EPI_BORDER   = "#7C3AED"   # violet — epistemic frame, distinct from red heatmap
    C_BLUE = "#1F77B4"           # correction / recovery

    # ── Token profile data: values approximate per-position clipped distillation signal ──
    heatmap_tokens = [
        ("Let", 0.024, "normal"),
        ("the", 0.020, "normal"),
        ("count", 0.032, "normal"),
        ("be", 0.026, "normal"),
        ("computed", 0.038, "normal"),
        ("by", 0.028, "normal"),
        ("choosing", 0.036, "normal"),
        ("two", 0.026, "normal"),
        ("positions.", 0.034, "normal"),
        ("This", 0.030, "normal"),
        ("gives", 0.032, "normal"),
        ("$48$", 0.190, "error"),
        ("arrangements.", 0.040, "normal"),
        ("Hmm", 0.172, "epistemic"),
        (",", 0.024, "normal"),
        ("wait", 0.184, "epistemic"),
        (",", 0.022, "normal"),
        ("that", 0.034, "normal"),
        ("seems", 0.156, "epistemic"),
        ("too", 0.038, "normal"),
        ("large.", 0.040, "normal"),
        ("Maybe", 0.168, "epistemic"),
        ("I", 0.026, "normal"),
        ("counted", 0.042, "normal"),
        ("each", 0.034, "normal"),
        ("pair", 0.030, "normal"),
        ("twice.", 0.036, "normal"),
        ("Actually", 0.160, "epistemic"),
        (",", 0.024, "normal"),
        ("divide", 0.036, "normal"),
        ("by", 0.026, "normal"),
        ("2", 0.030, "normal"),
        ("to", 0.024, "normal"),
        ("get", 0.028, "normal"),
        ("$24$", 0.032, "correction"),
        ("arrangements.", 0.038, "normal"),
        ("So", 0.030, "normal"),
        ("the", 0.024, "normal"),
        ("answer", 0.034, "normal"),
        ("is", 0.026, "normal"),
        ("$24$.", 0.034, "normal"),
    ]

    # ── Figure layout ─────────────────────────────────────────────
    fig = plt.figure(figsize=(6.8, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 1.0], hspace=0.40)

    # ── Top panel: token-level signal heatmap ─────────────────────
    ax_top = fig.add_subplot(gs[0])
    # Sequential white → red: high signal reads as visual "pressure"
    cmap = LinearSegmentedColormap.from_list(
        "kl_whites_reds",
        ["#FFFFFF", "#FFE5E5", "#F5A8A8", "#D62728"],
    )
    norm = Normalize(vmin=0.0, vmax=0.20)

    def token_width(token):
        return max(0.50, 0.17 * len(token) + 0.26)

    x0, y0 = 0.18, 4.05
    x, y = x0, y0
    max_x = 10.85
    row_h = 0.74
    box_h = 0.54
    for token, kl, kind in heatmap_tokens:
        w = token_width(token)
        if x + w > max_x:
            x = x0
            y -= row_h

        edgecolor = "#E0E0E0"
        linewidth = 0.6
        if kind == "epistemic":
            edgecolor = C_EPI_BORDER
            linewidth = 2.2
        elif kind == "error":
            edgecolor = C_ERROR_BORDER
            linewidth = 2.0
        elif kind == "correction":
            edgecolor = C_BLUE
            linewidth = 1.6

        patch = FancyBboxPatch(
            (x, y), w, box_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=cmap(norm(kl)),
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax_top.add_patch(patch)
        # Auto-pick label color based on cell darkness
        rgba = cmap(norm(kl))
        text_color = "#FFFFFF" if (rgba[0] + rgba[1] + rgba[2]) / 3 < 0.55 else "#1A1A1A"
        ax_top.text(
            x + w / 2, y + box_h / 2, token,
            ha="center", va="center",
            fontsize=8.6,
            fontfamily="monospace",
            fontweight="bold",
            color=text_color,
        )
        x += w + 0.07

    # Colorbar — sized to be readable
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = ax_top.inset_axes([0.74, 0.92, 0.24, 0.06])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cb.outline.set_visible(False)
    cax.text(-0.08, 0.5, "Low signal", ha="right", va="center",
             fontsize=8.5, fontweight="bold", color="#444444",
             transform=cax.transAxes)
    cax.text(1.08, 0.5, "High signal", ha="left", va="center",
             fontsize=8.5, fontweight="bold", color="#D62728",
             transform=cax.transAxes)

    # Inline category legend pushed to top-left
    legend_items = [
        ("epistemic", C_EPI_BORDER, 0.22, 1.40),
        ("error",     C_ERROR_BORDER, 1.85, 0.95),
        ("correction", C_BLUE,      3.00, 1.40),
    ]
    for label, edgecolor, lx, lw in legend_items:
        patch = FancyBboxPatch(
            (lx, 0.42), lw, box_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#FFFFFF",
            edgecolor=edgecolor,
            linewidth=2.2 if label != "correction" else 1.8,
        )
        ax_top.add_patch(patch)
        ax_top.text(
            lx + lw / 2,
            0.42 + box_h / 2,
            label,
            ha="center", va="center",
            fontsize=8.0,
            color="#333333",
            fontweight="bold",
        )

    ax_top.set_xlim(0, 11.1)
    ax_top.set_ylim(0.10, 5.85)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title("Dense self-distillation conflates correction with epistemic suppression",
                     fontsize=11, fontweight="semibold", pad=12)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # ── Bottom panel: bar chart ───────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])

    categories = ["Normal", "Error", "Epistemic"]
    mean_kls = [0.013, 0.147, 0.123]
    colors_bar = [C_GRAY, C_ERROR_BORDER, C_EPI_BORDER]
    x_pos = np.arange(len(categories))

    bars = ax_bar.bar(x_pos, mean_kls, 0.58, color=colors_bar,
                      edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars, mean_kls):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                    f"{val:.3f}", ha="center", fontsize=13,
                    fontweight="bold", color="#1A1A1A")

    # Bracket between Error and Epistemic
    bracket_y = 0.205
    ax_bar.plot([1, 1, 2, 2],
                [bracket_y, bracket_y + 0.012, bracket_y + 0.012, bracket_y],
                color="#666666", linewidth=1.3, clip_on=False)
    ax_bar.text(1.5, bracket_y + 0.030, "Comparable mean signal",
                ha="center", fontsize=11.5, color="#333333", fontweight="bold")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(categories, fontsize=13)
    ax_bar.set_ylabel("Mean token-level signal", fontsize=13)
    ax_bar.tick_params(axis="y", labelsize=11)
    ax_bar.set_ylim(0, 0.275)
    ax_bar.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.92, hspace=0.40)
    path = os.path.join(OUT, "fig_token_kl_strip.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures (legend inside, tight suptitle)...")
    fig_aime_curves("15b")
    fig_aime_curves("7b")
    fig_epi_len_dynamics("15b")
    fig_epi_len_dynamics("7b")
    fig_combined_4panel()
    fig_ref_context_probe()
    fig_prob_probe()
    fig_summary_delta("15b")
    fig_summary_delta("7b")
    fig_kl_concentration()
    fig_overview_naive()
    fig_overview_masked()
    fig_token_kl_strip()
    print(f"Done → {OUT}/")
