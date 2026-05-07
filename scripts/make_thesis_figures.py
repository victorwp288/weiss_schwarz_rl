#!/usr/bin/env python3
"""Build clean thesis figures from copied Vast evaluation artifacts."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "vast_artifacts"
OUT = ROOT / "thesis_figures_final"

BLUE = "#2F6F9F"
TEAL = "#2A9D8F"
AMBER = "#D89C27"
RED = "#C75D57"
PURPLE = "#7A5BA6"
GRAY = "#5F6470"
LIGHT_GRID = "#D8DCE2"


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def savefig(name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close()


def short_policy(label: str) -> str:
    if label == "B0 RandomLegal":
        return "B0 random"
    if label == "B1 NoLeague baseline":
        return "B1 no-league"
    if label == "B2 HeuristicPublic":
        return "B2 heuristic"
    if label == "B3 HeuristicPublicAggro":
        return "B3 aggro"
    if label == "B4 HeuristicPublicControl":
        return "B4 control"
    if label == "policy_000021":
        return "main p21"
    if "policy_000016" in label:
        return "legacy p16"
    if "policy_000015" in label:
        return "legacy p15"
    if "policy_000014" in label:
        return "legacy p14"
    if "policy_000012" in label:
        return "legacy p12"
    if "policy_000011" in label:
        return "legacy p11"
    return label


def display_policy(label: str) -> str:
    if label == "B0 RandomLegal":
        return "B0 RandomLegal"
    if label == "B1 NoLeague baseline":
        return "B1 NoLeague baseline"
    if label == "B2 HeuristicPublic":
        return "B2 HeuristicPublic"
    if label == "B3 HeuristicPublicAggro":
        return "B3 HeuristicPublicAggro"
    if label == "B4 HeuristicPublicControl":
        return "B4 HeuristicPublicControl"
    if "policy_000016" in label:
        return "Legacy p16"
    if "policy_000015" in label:
        return "Legacy p15"
    if "policy_000014" in label:
        return "Legacy p14"
    if "policy_000012" in label:
        return "Legacy p12"
    if "policy_000011" in label:
        return "Legacy p11"
    return label


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def targeted_rows():
    core64_path = ARTIFACTS / "main" / "p21_core_legacy_confirm64_summary.json"
    data = load_json(core64_path if core64_path.exists() else ARTIFACTS / "main" / "targeted_confirm_summary.json")
    rows = [r for r in data["rows"] if r["focal_policy_id"] == "policy_000021"]

    b1_legacy_path = ARTIFACTS / "main" / "p21_b1_legacy_confirm64_summary.json"
    if not b1_legacy_path.exists():
        b1_legacy_path = ARTIFACTS / "main" / "p21_b1_legacy_confirm32_summary.json"
    if b1_legacy_path.exists():
        for replacement in load_json(b1_legacy_path)["rows"]:
            rows = [
                replacement if row["opponent_policy_id"] == replacement["opponent_policy_id"] else row
                for row in rows
            ]

    for row_path in sorted((ARTIFACTS / "main" / "confirm64_rows").glob("p21_vs_*_summary.json")):
        payload = load_json(row_path)
        summary = payload["summary"]
        uncertainty = payload["uncertainty"]
        replacement = {
            "focal_policy_id": payload["focal_policy_id"],
            "opponent_policy_id": payload["opponent_policy_id"],
            "paired_seeds": uncertainty.get("paired_seed_count", payload.get("observed_paired_seeds")),
            "games": summary["games"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "draws": summary["draws"],
            "mean": uncertainty["mean"],
            "ci_low": uncertainty["ci_low"],
            "ci_high": uncertainty["ci_high"],
            "prob_gt_half": uncertainty["prob_gt_half"],
            "truncations": summary["truncations"],
            "engine_errors": summary["engine_errors"],
        }
        rows = [
            replacement if row["opponent_policy_id"] == replacement["opponent_policy_id"] else row
            for row in rows
        ]

    b3b4_path = ARTIFACTS / "main" / "p21_b3b4_loopfix_confirm64_summary.json"
    if b3b4_path.exists():
        b3b4_rows = load_json(b3b4_path)["rows"]
        anchor_rows = rows[:3] + b3b4_rows
        league_rows = rows[3:]
        rows = anchor_rows + league_rows

    if core64_path.exists() or b1_legacy_path.name.endswith("confirm64_summary.json"):
        evidence_text = "confirm64 rows"
    elif b1_legacy_path.exists():
        evidence_text = "confirm32 and confirm64 rows"
    else:
        evidence_text = "mixed 32-game and confirm64 rows"
    return rows, evidence_text


def format_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_targeted_robustness():
    rows, evidence_text = targeted_rows()

    # Keep anchors first, then legacy league opponents.
    labels = [short_policy(r["opponent_policy_id"]) for r in rows]
    means = np.array([100 * r["mean"] for r in rows])
    lows = np.array([100 * r["ci_low"] for r in rows])
    highs = np.array([100 * r["ci_high"] for r in rows])
    wins = [r["wins"] for r in rows]
    games = [r["games"] for r in rows]
    y = np.arange(len(rows))
    colors = [TEAL if i < 5 else BLUE for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.barh(y, means, color=colors, height=0.64)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="none",
        ecolor="#1F2328",
        capsize=4,
        linewidth=1.4,
    )
    ax.axvline(50, color="#1F2328", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 106)
    ax.set_xlabel("Win rate (%)")
    ax.set_title(
        f"Main League GRU targeted evaluation\npolicy_000021; {evidence_text}",
        fontsize=11.8,
        pad=14,
    )
    for yi, m, w, g in zip(y, means, wins, games):
        ax.text(101.5, yi, f"{w}/{g}", va="center", ha="left", fontsize=9.5)
        if m < 98:
            ax.text(m - 1.2, yi, f"{m:.1f}%", va="center", ha="right", color="white", fontsize=9.5)
    format_axes(ax)
    fig.subplots_adjust(bottom=0.12, top=0.84, left=0.20, right=0.93)
    savefig("fig_main_targeted_robustness")


def figure_b3b4_validity():
    rows = load_json(ARTIFACTS / "main" / "p21_b3b4_loopfix_confirm64_summary.json")["rows"]
    labels = [short_policy(row["opponent_policy_id"]) for row in rows]
    win_rates = np.array([100 * row["mean"] for row in rows])
    valid_rates = np.array([100 * (row["games"] - row.get("truncations", 0)) / row["games"] for row in rows])
    x = np.arange(len(rows))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars_valid = ax.bar(x - width / 2, valid_rates, width=width, color=GRAY, label="completed games")
    bars_win = ax.bar(x + width / 2, win_rates, width=width, color=TEAL, label="p21 win rate")
    for bar, row in zip(bars_valid, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{row['games'] - row.get('truncations', 0)}/{row['games']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, row in zip(bars_win, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{row['wins']}/{row['games']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(50, color="#1F2328", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 116)
    ax.set_ylabel("Rate (%)")
    ax.set_title("B3/B4 fixed-opponent validation", fontsize=13.5, pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, fontsize=9)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.23, top=0.86, left=0.12, right=0.98)
    savefig("fig_b3b4_fixed_validation")


def figure_b3b4_seat_balance():
    specs = [
        ("B3 aggro", ARTIFACTS / "main" / "diagnostics" / "p21_vs_b3_diagnostics.json"),
        ("B4 control", ARTIFACTS / "main" / "diagnostics" / "p21_vs_b4_diagnostics.json"),
    ]
    labels, seat0, seat1 = [], [], []
    for label, path in specs:
        if not path.exists():
            return
        data = load_json(path)
        p21 = data["policy_breakdown"]["policy_000021"]
        labels.append(label)
        seat0.append(100 * p21["wins_as_seat0"] / p21["games_as_seat0"])
        seat1.append(100 * p21["wins_as_seat1"] / p21["games_as_seat1"])

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    b0 = ax.bar(x - width / 2, seat0, width=width, color=BLUE, label="p21 first seat")
    b1 = ax.bar(x + width / 2, seat1, width=width, color=AMBER, label="p21 second seat")
    for bars in (b0, b1):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.8,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(50, color="#1F2328", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("B3/B4 seat-swapped robustness", fontsize=13.5, pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, fontsize=9)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.23, top=0.86, left=0.12, right=0.98)
    savefig("fig_b3b4_seat_balance")


def figure_p21_seat_advantage():
    path = ARTIFACTS / "main" / "seat_diagnostics" / "p21_headline_confirm64_seat_diagnostics.json"
    if not path.exists():
        path = ARTIFACTS / "main" / "seat_diagnostics" / "p21_headline_seat_diagnostics.json"
    if not path.exists():
        return
    rows = load_json(path)
    labels = [short_policy(row["opponent"]) for row in rows]
    first = np.array([100 * row["wins_as_first"] / row["games_as_first"] for row in rows], dtype=float)
    second = np.array([100 * row["wins_as_second"] / row["games_as_second"] for row in rows], dtype=float)
    diff = second - first
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = [TEAL if value >= 0 else RED for value in diff]
    ax.barh(y, diff, color=colors, height=0.62)
    ax.axvline(0, color="#1F2328", linewidth=1.1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Second-seat minus first-seat win rate (percentage points)")
    first_wins = sum(row["wins_as_first"] for row in rows)
    first_games = sum(row["games_as_first"] for row in rows)
    second_wins = sum(row["wins_as_second"] for row in rows)
    second_games = sum(row["games_as_second"] for row in rows)
    ax.set_title(
        f"Seat sensitivity of policy_000021\naggregate: first {first_wins}/{first_games}, second {second_wins}/{second_games}",
        fontsize=12.8,
        pad=12,
    )
    for yi, value, row in zip(y, diff, rows):
        label = f"{value:+.1f} pp"
        xpos = value + (0.6 if value >= 0 else -0.6)
        ha = "left" if value >= 0 else "right"
        ax.text(xpos, yi, label, va="center", ha=ha, fontsize=9.5)
    max_abs = max(8, float(np.nanmax(np.abs(diff))) + 3)
    ax.set_xlim(-max_abs, max_abs)
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.16, top=0.82, left=0.21, right=0.97)
    savefig("fig_p21_seat_advantage")


def figure_close_legacy_stress():
    path64 = ARTIFACTS / "main" / "p21_b1_legacy_confirm64_summary.json"
    path128 = ARTIFACTS / "main" / "p21_p15_p16_confirm128_summary.json"
    if not (path64.exists() and path128.exists()):
        return
    rows64 = {
        row["opponent_policy_id"]: row
        for row in load_json(path64)["rows"]
        if "policy_000015" in row["opponent_policy_id"] or "policy_000016" in row["opponent_policy_id"]
    }
    rows128 = {row["opponent_policy_id"]: row for row in load_json(path128)["rows"]}
    labels = ["legacy p15", "legacy p16"]
    keys = [
        next(key for key in rows128 if "policy_000015" in key),
        next(key for key in rows128 if "policy_000016" in key),
    ]
    vals64 = np.array([100 * rows64[key]["mean"] for key in keys])
    vals128 = np.array([100 * rows128[key]["mean"] for key in keys])
    lows128 = np.array([100 * rows128[key]["ci_low"] for key in keys])
    highs128 = np.array([100 * rows128[key]["ci_high"] for key in keys])
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    b64 = ax.bar(x - width / 2, vals64, width=width, color=BLUE, label="confirm64")
    b128 = ax.bar(x + width / 2, vals128, width=width, color=TEAL, label="confirm128")
    ax.errorbar(
        x + width / 2,
        vals128,
        yerr=np.vstack([vals128 - lows128, highs128 - vals128]),
        fmt="none",
        ecolor="#1F2328",
        capsize=4,
        linewidth=1.2,
    )
    for bars, rows in ((b64, [rows64[key] for key in keys]), (b128, [rows128[key] for key in keys])):
        for bar, row in zip(bars, rows):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{int(row['wins'])}/{int(row['games'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(50, color="#1F2328", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, labels)
    ax.set_ylim(44, 58)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Close legacy rows: confirm128 stress check", fontsize=13.5, pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2, fontsize=9)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.25, top=0.86, left=0.12, right=0.98)
    savefig("fig_close_legacy_stress")


def dev_points(run: str):
    data = load_json(ARTIFACTS / run / "dev_eval_summaries.json")
    points = []
    for item in data.values():
        points.append(
            (
                item["update_count"],
                100 * item["aggregate_score"],
                100 * item["anchor_scores"].get("B1 NoLeague baseline", math.nan),
            )
        )
    return np.array(sorted(points))


def figure_anchor_retention():
    specs = [
        ("main", "main league GRU", BLUE),
        ("exp028", "no B1 lane", RED),
        ("exp029", "weak B1 mix", AMBER),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for run, label, color in specs:
        pts = dev_points(run)
        ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=4, lw=1.8, label=f"{label}: anchor avg", color=color)
        ax.plot(pts[:, 0], pts[:, 2], marker="s", ms=3.5, lw=1.3, linestyle="--", color=color, alpha=0.75)
    ax.set_title("Periodic dev evaluation on fixed anchors", fontsize=15, pad=12)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Win rate / aggregate score (%)")
    ax.set_ylim(65, 102)
    ax.axhline(80, color="#1F2328", linestyle=":", lw=1.0)
    ax.text(0.995, 0.02, "Dashed lines: B1 no-league retention", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=GRAY)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    format_axes(ax)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
    fig.subplots_adjust(bottom=0.16, top=0.88, left=0.10, right=0.98)
    savefig("fig_anchor_retention")


def figure_core_matrix():
    data = load_json(ARTIFACTS / "main" / "final_summary.json")
    labels = [short_policy(x) for x in data["policy_ids"]]
    values = np.array(data["matrices"]["mean"]["values"]) * 100

    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    im = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text_color = "white" if val <= 25 or val >= 75 else "#1F2328"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=9)
    ax.set_title("Fast final matrix sanity check", fontsize=14.5, pad=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Win rate (%)")
    fig.subplots_adjust(bottom=0.28, top=0.88, left=0.20, right=0.93)
    savefig("fig_fast_matrix_sanity")


def final_mean(path: Path, focal: str, opponent: str):
    data = load_json(path)
    for row in data["rows"]:
        if row["focal_policy_id"] == focal and row["opponent_policy_id"] == opponent:
            return row["mean"], row["wins"], row["games"]
    return None, None, None


def fixed_baseline_rows(path: Path):
    data = load_json(path)
    policy_ids = data["policy_ids"]
    focal_idx = len(policy_ids) - 1
    mean_matrix = data["matrices"]["mean"]["values"]
    win_matrix = data["matrices"]["wins"]["values"]
    game_matrix = data["matrices"]["games"]["values"]
    out = {}
    for opp in ("B0 RandomLegal", "B2 HeuristicPublic"):
        opp_idx = policy_ids.index(opp)
        out[short_policy(opp)] = (
            mean_matrix[focal_idx][opp_idx],
            win_matrix[focal_idx][opp_idx],
            game_matrix[focal_idx][opp_idx],
        )
    return out


def figure_baselines():
    main = ARTIFACTS / "main" / "targeted_confirm_summary.json"
    groups_full = {
        "main p21": {
            "B0 random": final_mean(main, "policy_000021", "B0 RandomLegal"),
            "B1 no-league": final_mean(main, "policy_000021", "B1 NoLeague baseline"),
            "B2 heuristic": final_mean(main, "policy_000021", "B2 HeuristicPublic"),
        },
        "No-GRU": fixed_baseline_rows(ARTIFACTS / "nogru" / "final_summary.json"),
        "PPO-lite": fixed_baseline_rows(ARTIFACTS / "ppo" / "final_summary.json"),
    }
    groups = {
        model: {opp: rows[opp] for opp in ("B0 random", "B2 heuristic")}
        for model, rows in groups_full.items()
    }
    opponents = ["B0 random", "B2 heuristic"]
    x = np.arange(len(groups))
    width = 0.28
    colors = [TEAL, AMBER]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, opp in enumerate(opponents):
        vals = []
        labels = []
        for model, rows in groups.items():
            mean, wins, games = rows.get(opp, (math.nan, None, None))
            vals.append(mean * 100 if mean == mean else np.nan)
            labels.append("" if wins is None else f"{wins}/{games}")
        bars = ax.bar(x + (k - 0.5) * width, vals, width=width, color=colors[k], label=opp)
        for bar, lab, val in zip(bars, labels, vals):
            if lab:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 2.0, lab, ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, list(groups.keys()))
    ax.set_ylim(0, 112)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Baseline sanity checks against fixed opponents", fontsize=13.5, pad=14)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.24, top=0.88, left=0.10, right=0.98)
    savefig("fig_baseline_sanity_fixed")

    labels = list(groups.keys())
    grid = np.full((len(labels), len(opponents)), np.nan)
    cell_text: list[list[str]] = []
    for i, model in enumerate(labels):
        row_text = []
        for j, opp in enumerate(opponents):
            mean, wins, games = groups[model].get(opp, (math.nan, None, None))
            if wins is None:
                row_text.append("n/a")
            else:
                grid[i, j] = mean * 100
                row_text.append(f"{100 * mean:.1f}%\n{wins}/{games}")
        cell_text.append(row_text)

    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad("#E7E9EE")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(opponents)), opponents)
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(opponents)):
            val = grid[i, j]
            txt = cell_text[i][j]
            color = "white" if val == val and val >= 70 else "#1F2328"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    ax.set_title("Fixed-opponent baseline checks", fontsize=13.5, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.70)
    cbar.set_label("Win rate (%)")
    fig.subplots_adjust(bottom=0.14, top=0.84, left=0.20, right=0.92)
    savefig("fig_baseline_fixed_grid")


def figure_training_diagnostic():
    path = ARTIFACTS / "main" / "training_metrics.jsonl"
    updates, losses, entropy = [], [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            update = item.get("update") or item.get("update_count")
            loss = item.get("total_loss", item.get("loss"))
            ent = item.get("entropy")
            if update is not None and loss is not None:
                updates.append(update)
                losses.append(loss)
                entropy.append(ent if ent is not None else math.nan)
    if not updates:
        return
    updates = np.array(updates)
    losses = np.array(losses, dtype=float)
    entropy = np.array(entropy, dtype=float)

    def smooth(y, window=21):
        if len(y) < window:
            return y
        kernel = np.ones(window) / window
        padded = np.pad(y, (window // 2, window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(updates, losses, color="#B9C2CF", lw=0.8, alpha=0.7, label="raw total loss")
    ax.plot(updates, smooth(losses), color=BLUE, lw=2.0, label="smoothed total loss")
    ax.set_title("Training optimization diagnostic", fontsize=14.5, pad=14)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Total loss")
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.16, top=0.88, left=0.10, right=0.98)
    savefig("fig_training_loss_diagnostic")


def write_main_table():
    rows, evidence_text = targeted_rows()
    caption = (
        "Targeted evaluation of the selected main League GRU model, "
        "policy\\_000021. All rows use 64 paired seeds (128 games) with "
        "seat-swapped evaluation."
        if evidence_text == "confirm64 rows"
        else "Targeted evaluation of the selected main League GRU model, policy\\_000021."
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:main-p21-targeted}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Opponent & Wins & Games & Win rate & Issues \\",
        r"\midrule",
    ]
    for row in rows:
        issues = int(row.get("truncations", 0)) + int(row.get("engine_errors", 0))
        lines.append(
            f"{latex_escape(display_policy(row['opponent_policy_id']))} & "
            f"{int(row['wins'])} & {int(row['games'])} & "
            f"{100 * float(row['mean']):.2f}\\% & {issues} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUT / "main_p21_results_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_section7_snippets():
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{thesis_figures_final/fig_main_targeted_robustness.pdf}
\caption{Targeted confirm64 evaluation of the selected main League GRU policy against fixed anchors, corrected B3/B4 public heuristics, and legacy league snapshots. Error bars show bootstrap confidence intervals over paired seeds.}
\label{fig:main-targeted-robustness}
\end{figure}

\input{thesis_figures_final/main_p21_results_table.tex}

\begin{figure}[t]
\centering
\includegraphics[width=0.82\linewidth]{thesis_figures_final/fig_b3b4_fixed_validation.pdf}
\caption{Validation of the corrected B3/B4 evaluation rows. The corrected aggressive/control heuristic opponents complete all games without truncation, and the selected policy remains above parity against both profiles.}
\label{fig:b3b4-fixed-validation}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.82\linewidth]{thesis_figures_final/fig_p21_seat_advantage.pdf}
\caption{Seat sensitivity diagnostic for the targeted confirm64 table. Evaluations are paired and seat-swapped, so the measured first/second-seat asymmetry is diagnostic rather than a confound in the reported win rates.}
\label{fig:p21-seat-sensitivity}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{thesis_figures_final/fig_close_legacy_stress.pdf}
\caption{Stress check for the two closest legacy neural rows. The rows remain slightly above parity at confirm128, but the margins are narrow and should be reported conservatively.}
\label{fig:close-legacy-stress}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.88\linewidth]{thesis_figures_final/fig_anchor_retention.pdf}
\caption{Periodic development evaluation for the main model and B1-pressure ablations. Solid lines show aggregate anchor performance; dashed lines show B1 no-league retention.}
\label{fig:anchor-retention}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{thesis_figures_final/fig_baseline_fixed_grid.pdf}
\caption{Fixed-opponent baseline sanity checks. No-GRU and PPO-lite were not evaluated against the full B1/league opponent set, so this figure is a fixed-anchor baseline comparison rather than a full league-robustness comparison.}
\label{fig:fixed-baseline-grid}
\end{figure}
"""
    (OUT / "section7_figure_snippets.tex").write_text(text, encoding="utf-8")


def write_captions():
    rows, evidence_text = targeted_rows()
    total_wins = sum(int(row["wins"]) for row in rows)
    total_games = sum(int(row["games"]) for row in rows)
    b3b4 = [row for row in rows if row["opponent_policy_id"].startswith("B3") or row["opponent_policy_id"].startswith("B4")]
    b3b4_wins = sum(int(row["wins"]) for row in b3b4)
    b3b4_games = sum(int(row["games"]) for row in b3b4)
    legacy = [row for row in rows if "policy_0000" in row["opponent_policy_id"]]
    legacy_wins = sum(int(row["wins"]) for row in legacy)
    legacy_games = sum(int(row["games"]) for row in legacy)
    b1 = next((row for row in rows if row["opponent_policy_id"] == "B1 NoLeague baseline"), None)
    b1_text = "not available" if b1 is None else f"{int(b1['wins'])}/{int(b1['games'])} ({100 * float(b1['mean']):.1f}%)"
    text = f"""# Thesis Figure Captions

## fig_main_targeted_robustness
Targeted evaluation of the selected main League GRU model (`policy_000021`) against fixed anchors, corrected B3/B4 public heuristics, and legacy league opponents. Evidence level: {evidence_text}. Overall targeted table: {total_wins}/{total_games} ({100 * total_wins / total_games:.1f}%). B1 no-league: {b1_text}. B3/B4 combined: {b3b4_wins}/{b3b4_games} ({100 * b3b4_wins / b3b4_games:.1f}%). Legacy neural subset: {legacy_wins}/{legacy_games} ({100 * legacy_wins / legacy_games:.1f}%).

## fig_anchor_retention
Periodic development evaluation on fixed anchors for the main model and B1-pressure ablations. Solid lines show aggregate anchor score; dashed lines show B1 no-league retention. Use this to argue that the selected model retains anchor strength while entering league training.

## fig_fast_matrix_sanity
Fast final matrix with 4 paired seeds per matchup. This is useful as a qualitative sanity check and figure for the artifact set, but it is not strong enough as the headline quantitative claim.

## fig_baseline_sanity_fixed
Fixed-opponent baseline sanity check against B0 and B2 only. B1 is intentionally excluded here because the No-GRU and PPO-lite matrices were produced against fixed non-neural opponents only; B1 evidence for the main model belongs in the targeted robustness figure.

## fig_baseline_fixed_grid
Table-style version of the fixed-opponent baseline sanity check against B0 and B2 only. This is clearer than bars because the No-GRU and PPO-lite B2 results are true 0/32 outcomes.

## fig_training_loss_diagnostic
Smoothed training loss diagnostic for the main run. Keep this out of the main results argument unless needed for transparency; actor-critic loss is noisy and evaluation win rates are more meaningful.

## fig_b3b4_fixed_validation
Validation figure for the corrected B3/B4 evaluation rows. It shows that all B3/B4 confirm64 games completed without truncation and that `policy_000021` wins 82/128 against B3 and 83/128 against B4.

## fig_b3b4_seat_balance
Seat-swapped B3/B4 robustness diagnostic. `policy_000021` remains above 50% both when moving first and when moving second, which supports using the paired-seat evaluation rather than a single-seat result.

## fig_p21_seat_advantage
Headline-table seat sensitivity diagnostic for `policy_000021`. Positive values mean the policy won more often from the second seat than the first seat for that opponent. The evaluations are paired and seat-swapped, so the seat split is a diagnostic rather than a confound.

## fig_close_legacy_stress
Stress check for the two closest legacy neural rows. Both p15 and p16 remain slightly above parity at confirm128 (`129/256` each), but their bootstrap intervals overlap 50%, so they should be described as narrow positive margins rather than decisive wins.
"""
    (OUT / "FIGURE_CAPTIONS.md").write_text(text)


def main():
    figure_targeted_robustness()
    figure_b3b4_validity()
    figure_b3b4_seat_balance()
    figure_p21_seat_advantage()
    figure_close_legacy_stress()
    figure_anchor_retention()
    figure_core_matrix()
    figure_baselines()
    figure_training_diagnostic()
    write_main_table()
    write_section7_snippets()
    write_captions()


if __name__ == "__main__":
    main()
