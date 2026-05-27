"""Figure and table exports for decision-time god-search evaluations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GodSearchFigurePaths:
    row_csv: Path
    group_summary_json: Path
    row_table_md: Path
    row_win_rates_png: Path
    row_win_rates_pdf: Path
    delta_wins_png: Path
    delta_wins_pdf: Path
    group_rates_png: Path
    group_rates_pdf: Path


@dataclass(frozen=True)
class MainSearchExtraFigurePaths:
    strength_ladder_png: Path
    strength_ladder_pdf: Path
    validation_progression_png: Path
    validation_progression_pdf: Path
    decision_changes_png: Path
    decision_changes_pdf: Path
    seat_balance_png: Path
    seat_balance_pdf: Path
    first_second_balance_png: Path
    first_second_balance_pdf: Path


def _safe_label(label: str) -> str:
    exact = {
        "B0 RandomLegal": "B0 Random Legal",
        "B1 NoLeague baseline": "B1 No-League",
        "B2 HeuristicPublic": "B2 Public heuristic",
        "B3 HeuristicPublicAggro": "B3 Aggro heuristic",
        "B4 HeuristicPublicControl": "B4 Control heuristic",
        "seed_c3aac2f9dc_policy_000001": "League policy 1",
        "seed_c3aac2f9dc_policy_000002": "League policy 2",
        "seed_c3aac2f9dc_checkpoint_000025": "Checkpoint u25",
        "seed_c3aac2f9dc_main_bestresponse_u25_devbest": "Best response u25",
        "seed_c3aac2f9dc_main_league_selected": "Imported main selected",
        "seed_c3aac2f9dc_policy_000003": "League policy 3",
        "seed_c3aac2f9dc_policy_000004": "League policy 4",
        "seed_c3aac2f9dc_policy_000005": "League policy 5",
    }
    if label in exact:
        return exact[label]
    return (
        label.replace("seed_c3aac2f9dc_", "")
        .replace("main_bestresponse_u25_devbest", "bestresponse_u25")
        .replace("main_league_selected", "main_selected")
    )


def _install_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#333333",
            "axes.edgecolor": "#555555",
            "font.size": 9,
            "legend.fontsize": 8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _row_records(compare_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = compare_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("compare payload must contain non-empty rows")
    records: list[dict[str, Any]] = []
    for row in rows:
        shared_games = int(row["shared_games"])
        baseline_wins = int(row["baseline_wins"])
        candidate_wins = int(row["candidate_wins"])
        opponent = str(row["opponent_policy_id"])
        group = str(row.get("group") or "")
        if not group:
            group = "fixed" if opponent.startswith("B") else "learned"
        records.append(
            {
                "group": group,
                "opponent_id": opponent,
                "opponent": _safe_label(opponent),
                "baseline_wins": baseline_wins,
                "candidate_wins": candidate_wins,
                "delta_wins": int(row["delta_wins"]),
                "shared_games": shared_games,
                "baseline_rate": baseline_wins / shared_games,
                "candidate_rate": candidate_wins / shared_games,
                "delta_rate": (candidate_wins - baseline_wins) / shared_games,
            }
        )
    return records


def _group_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, dict[str, float | int]] = {}
    for group in ("fixed", "learned", "all"):
        subset = records if group == "all" else [row for row in records if row["group"] == group]
        if not subset:
            continue
        baseline_wins = sum(int(row["baseline_wins"]) for row in subset)
        candidate_wins = sum(int(row["candidate_wins"]) for row in subset)
        games = sum(int(row["shared_games"]) for row in subset)
        groups[group] = {
            "baseline_wins": baseline_wins,
            "candidate_wins": candidate_wins,
            "delta_wins": candidate_wins - baseline_wins,
            "games": games,
            "baseline_rate": baseline_wins / games,
            "candidate_rate": candidate_wins / games,
            "delta_rate": (candidate_wins - baseline_wins) / games,
        }
    return groups


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "opponent_id",
        "opponent",
        "baseline_wins",
        "candidate_wins",
        "delta_wins",
        "shared_games",
        "baseline_rate",
        "candidate_rate",
        "delta_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_md_table(
    path: Path, records: list[dict[str, Any]], group_summary: dict[str, dict[str, float | int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Main Search Confirm256 Row Table",
        "",
        "| Group | Opponent | Selected wins | K4 search wins | Delta | K4 win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {group} | {opponent} | {baseline_wins}/{games} | {candidate_wins}/{games} | {delta:+d} | {rate:.3f} |".format(
                group=row["group"],
                opponent=row["opponent"],
                baseline_wins=row["baseline_wins"],
                candidate_wins=row["candidate_wins"],
                games=row["shared_games"],
                delta=row["delta_wins"],
                rate=row["candidate_rate"],
            )
        )
    lines.extend(["", "## Group Summary", "", "| Group | Selected | K4 search | Delta |", "|---|---:|---:|---:|"])
    for group in ("fixed", "learned", "all"):
        summary = group_summary[group]
        lines.append(
            "| {group} | {base}/{games} ({base_rate:.3f}) | {cand}/{games} ({cand_rate:.3f}) | {delta:+d} |".format(
                group=group,
                base=int(summary["baseline_wins"]),
                cand=int(summary["candidate_wins"]),
                games=int(summary["games"]),
                base_rate=float(summary["baseline_rate"]),
                cand_rate=float(summary["candidate_rate"]),
                delta=int(summary["delta_wins"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_bar_figure(
    path_png: Path,
    path_pdf: Path,
    *,
    labels: list[str],
    series: list[tuple[str, list[float]]],
    ylabel: str,
    title: str,
    value_format: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    _install_plot_style()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(len(labels)))
    width = 0.78 / max(len(series), 1)
    fig_width = max(7.2, len(labels) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    colors = ["#356ac3", "#e67e22", "#2f9e62"]
    for idx, (name, values) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(
            [value + offset for value in x],
            values,
            width=width,
            label=name,
            color=colors[idx % len(colors)],
        )
        if value_format is not None:
            ax.bar_label(bars, labels=[value_format.format(value) for value in values], padding=3, fontsize=8)
    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    if all(0.0 <= value <= 1.05 for _, values in series for value in values):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, 1.04)
    ax.grid(axis="y", color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=len(series))
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def _save_horizontal_bar_figure(
    path_png: Path,
    path_pdf: Path,
    *,
    labels: list[str],
    series: list[tuple[str, list[float]]],
    xlabel: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    value_format: str | None = None,
    zero_line: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    _install_plot_style()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)

    y = list(range(len(labels)))
    height = 0.78 / max(len(series), 1)
    fig_height = max(4.2, len(labels) * 0.36 + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    colors = ["#356ac3", "#e67e22", "#2f9e62"]
    for idx, (name, values) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * height
        bars = ax.barh(
            [value + offset for value in y],
            values,
            height=height,
            label=name,
            color=colors[idx % len(colors)],
        )
        if value_format is not None:
            ax.bar_label(
                bars,
                labels=[value_format.format(value) for value in values],
                padding=3,
                fontsize=8,
            )
    ax.set_title(title, pad=20 if len(series) > 1 else 10)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if all(0.0 <= value <= 1.05 for _, values in series for value in values):
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if zero_line:
        ax.axvline(0.0, color="#555555", linewidth=0.9)
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncols=len(series))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    else:
        fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def write_god_search_figures(
    *, compare_json: Path, out_dir: Path, figure_prefix: str = "god_search_k4_confirm256"
) -> GodSearchFigurePaths:
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    records = _row_records(payload)
    groups = _group_summary(records)
    data_dir = out_dir / "data"
    paper_dir = out_dir / "paper"
    paths = GodSearchFigurePaths(
        row_csv=data_dir / f"{figure_prefix}_rows.csv",
        group_summary_json=data_dir / f"{figure_prefix}_group_summary.json",
        row_table_md=data_dir / f"{figure_prefix}_row_table.md",
        row_win_rates_png=paper_dir / f"{figure_prefix}_row_win_rates.png",
        row_win_rates_pdf=paper_dir / f"{figure_prefix}_row_win_rates.pdf",
        delta_wins_png=paper_dir / f"{figure_prefix}_delta_wins.png",
        delta_wins_pdf=paper_dir / f"{figure_prefix}_delta_wins.pdf",
        group_rates_png=paper_dir / f"{figure_prefix}_group_rates.png",
        group_rates_pdf=paper_dir / f"{figure_prefix}_group_rates.pdf",
    )

    _write_csv(paths.row_csv, records)
    paths.group_summary_json.parent.mkdir(parents=True, exist_ok=True)
    paths.group_summary_json.write_text(json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md_table(paths.row_table_md, records, groups)

    row_labels = [_safe_label(str(row["opponent"])) for row in records]
    _save_horizontal_bar_figure(
        paths.row_win_rates_png,
        paths.row_win_rates_pdf,
        labels=row_labels,
        series=[
            ("Main without search", [float(row["baseline_rate"]) for row in records]),
            ("Main with search", [float(row["candidate_rate"]) for row in records]),
        ],
        xlabel="Win rate",
        title="Confirm256 win rates by opponent",
        xlim=(0.0, 1.05),
    )
    _save_horizontal_bar_figure(
        paths.delta_wins_png,
        paths.delta_wins_pdf,
        labels=row_labels,
        series=[("Delta wins", [float(row["delta_wins"]) for row in records])],
        xlabel="Paired win delta",
        title="Search-enhanced main paired win deltas",
    )

    group_labels = ["Fixed B0-B4+B1", "Learned/hard negatives", "All rows"]
    group_keys = ["fixed", "learned", "all"]
    _save_bar_figure(
        paths.group_rates_png,
        paths.group_rates_pdf,
        labels=group_labels,
        series=[
            ("Main without search", [float(groups[key]["baseline_rate"]) for key in group_keys]),
            ("Main with search", [float(groups[key]["candidate_rate"]) for key in group_keys]),
        ],
        ylabel="Win rate",
        title="Confirm256 group win rates",
        value_format="{:.1%}",
    )
    return paths


def _save_strength_ladder(
    path_png: Path,
    path_pdf: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    _install_plot_style()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["model"]) for row in rows]
    metrics = [
        ("fixed", "Fixed B0-B4+B1", "#356ac3"),
        ("learned", "Learned and hard negatives", "#2f9e62"),
        ("all", "All rows", "#e67e22"),
    ]
    y_base = list(range(len(labels)))
    height = 0.22
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for metric_idx, (key, name, color) in enumerate(metrics):
        offset = (metric_idx - 1) * height
        y_values: list[float] = []
        x_values: list[float] = []
        for idx, row in enumerate(rows):
            value = row.get(key)
            if value is None:
                continue
            y_values.append(float(idx) + offset)
            x_values.append(float(value))
        bars = ax.barh(y_values, x_values, height=height, color=color, label=name)
        ax.bar_label(bars, labels=[_percent(value) for value in x_values], padding=3, fontsize=8)

    ax.axvline(0.5, color="#555555", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Win rate")
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Main model strength ladder")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def _save_validation_progression(
    path_png: Path,
    path_pdf: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    _install_plot_style()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)

    stage_labels = {
        "K3 confirm64": "K3 64",
        "K4 confirm64": "K4 64",
        "K4 confirm128": "K4 128",
        "K4 confirm256": "K4 256",
    }
    labels = [stage_labels.get(str(row["stage"]), str(row["stage"])) for row in rows]
    x = list(range(len(labels)))
    series = [
        ("fixed", "Fixed B0-B4+B1", "#356ac3", 0.006),
        ("learned", "Learned and hard negatives", "#2f9e62", -0.006),
        ("all", "All rows", "#e67e22", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    all_values: list[float] = []
    for key, name, color, text_offset in series:
        values = [float(row[key]) for row in rows]
        all_values.extend(values)
        ax.plot(x, values, marker="o", linewidth=2.0, markersize=4.5, color=color, label=name)
        for point_x, value in zip(x, values, strict=True):
            ax.text(point_x, value + text_offset, _percent(value), ha="center", va="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(max(0.0, min(all_values) - 0.04), min(1.0, max(all_values) + 0.04))
    ax.set_ylabel("Win rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Search validation progression")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def _save_decision_changes(
    path_png: Path,
    path_pdf: Path,
    *,
    payload: dict[str, Any],
) -> None:
    labels = [str(label) for label in payload["labels"]]
    deltas = [float(value) for value in payload["delta_wins"]]
    max_delta = max(deltas) if deltas else 0.0
    _save_horizontal_bar_figure(
        path_png,
        path_pdf,
        labels=labels,
        series=[("Additional wins", deltas)],
        xlabel="Additional wins over no-search main (out of 512 games)",
        title="Search-converted paired wins",
        xlim=(0.0, max_delta + 18.0),
        value_format="{:.0f}",
    )


def _save_seat_balance(
    path_png: Path,
    path_pdf: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _install_plot_style()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["label"]) for row in rows]
    values = [float(row["delta_pp"]) for row in rows]
    y = list(range(len(labels)))
    max_abs = max(abs(value) for value in values) if values else 1.0
    x_pad = max(1.5, max_abs * 0.28)
    colors = ["#2f9e62" if value > 0 else "#c84e4e" if value < 0 else "#9a9a9a" for value in values]

    fig, ax = plt.subplots(figsize=(8.2, max(4.4, len(labels) * 0.36 + 1.2)))
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.axvline(0.0, color="#555555", linewidth=0.9)
    ax.set_xlim(-(max_abs + x_pad), max_abs + x_pad)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seat 1 minus seat 0 win rate (percentage points)")
    ax.set_title("Confirm256 seat balance")
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        y_pos = bar.get_y() + bar.get_height() / 2
        if abs(value) < 0.05:
            ax.text(0.18, y_pos, "0.0 pp", ha="left", va="center", fontsize=8)
            continue
        text_x = value + (0.25 if value > 0 else -0.25)
        ha = "left" if value > 0 else "right"
        ax.text(text_x, y_pos, f"{value:+.1f} pp", ha=ha, va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def _save_first_second_balance(
    path_png: Path,
    path_pdf: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _install_plot_style()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    path_pdf.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row["label"]) for row in rows]
    values = [float(row["first_minus_second_pp"]) for row in rows]
    y = list(range(len(labels)))
    max_abs = max(abs(value) for value in values) if values else 1.0
    x_pad = max(1.5, max_abs * 0.28)
    colors = ["#356ac3" if value > 0 else "#c84e4e" if value < 0 else "#9a9a9a" for value in values]

    fig, ax = plt.subplots(figsize=(8.2, max(4.4, len(labels) * 0.36 + 1.2)))
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.axvline(0.0, color="#555555", linewidth=0.9)
    ax.set_xlim(-(max_abs + x_pad), max_abs + x_pad)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("First player minus second player win rate (percentage points)")
    ax.set_title("Confirm256 first-player balance")
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        y_pos = bar.get_y() + bar.get_height() / 2
        if abs(value) < 0.05:
            ax.text(0.18, y_pos, "0.0 pp", ha="left", va="center", fontsize=8)
            continue
        text_x = value + (0.25 if value > 0 else -0.25)
        ha = "left" if value > 0 else "right"
        ax.text(text_x, y_pos, f"{value:+.1f} pp", ha=ha, va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def write_main_search_extra_figures(
    *,
    data_dir: Path,
    paper_dir: Path,
    figure_prefix: str = "main_search",
) -> MainSearchExtraFigurePaths:
    paths = MainSearchExtraFigurePaths(
        strength_ladder_png=paper_dir / f"{figure_prefix}_strength_ladder.png",
        strength_ladder_pdf=paper_dir / f"{figure_prefix}_strength_ladder.pdf",
        validation_progression_png=paper_dir / f"{figure_prefix}_validation_progression.png",
        validation_progression_pdf=paper_dir / f"{figure_prefix}_validation_progression.pdf",
        decision_changes_png=paper_dir / f"{figure_prefix}_decision_changes.png",
        decision_changes_pdf=paper_dir / f"{figure_prefix}_decision_changes.pdf",
        seat_balance_png=paper_dir / f"{figure_prefix}_seat_balance.png",
        seat_balance_pdf=paper_dir / f"{figure_prefix}_seat_balance.pdf",
        first_second_balance_png=paper_dir / f"{figure_prefix}_first_second_balance.png",
        first_second_balance_pdf=paper_dir / f"{figure_prefix}_first_second_balance.pdf",
    )
    strength_rows = json.loads((data_dir / f"{figure_prefix}_strength_ladder.json").read_text(encoding="utf-8"))
    validation_rows = json.loads(
        (data_dir / f"{figure_prefix}_validation_progression.json").read_text(encoding="utf-8")
    )
    decision_payload = json.loads((data_dir / f"{figure_prefix}_decision_changes.json").read_text(encoding="utf-8"))
    seat_rows = json.loads((data_dir / f"{figure_prefix}_seat_balance.json").read_text(encoding="utf-8"))
    first_second_rows = json.loads(
        (data_dir / f"{figure_prefix}_first_second_balance.json").read_text(encoding="utf-8")
    )

    _save_strength_ladder(paths.strength_ladder_png, paths.strength_ladder_pdf, rows=strength_rows)
    _save_validation_progression(
        paths.validation_progression_png,
        paths.validation_progression_pdf,
        rows=validation_rows,
    )
    _save_decision_changes(paths.decision_changes_png, paths.decision_changes_pdf, payload=decision_payload)
    _save_seat_balance(paths.seat_balance_png, paths.seat_balance_pdf, rows=seat_rows)
    _save_first_second_balance(
        paths.first_second_balance_png,
        paths.first_second_balance_pdf,
        rows=first_second_rows,
    )
    return paths


__all__ = [
    "GodSearchFigurePaths",
    "MainSearchExtraFigurePaths",
    "write_god_search_figures",
    "write_main_search_extra_figures",
]
