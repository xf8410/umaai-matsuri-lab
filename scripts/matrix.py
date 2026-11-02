#!/usr/bin/env python3
"""骏大祭手写训练员 · 批量矩阵跑局 + 全指标采集

一局一 seed，采集：评分 / 档次 / 五维分列 + 合计 / PT / 稽古设施（演目）Lv 与上限 /
守门链各条命中次数 / 教诲融合（= buff 替换）逐条明细 / 助力支撑获得与消耗 /
追加稽古掷中·兑现·作废。

跑局靠仓内 bin `matsuri`（数据根 `data/` 为代码级配置，必须从仓根运行）。
配置走引擎覆盖层约定：仓根写 `ai_config.toml`（只给 `[sim]`），与 `data/ai_config.toml` 合并。

用法：
    python3 scripts/matrix.py --count 100 --start-seed 61444 --out artifacts/matrix
    python3 scripts/matrix.py --count 40 --start-seed 61444 --cards 303174,303114,303004,303224,303204,303234
"""

from __future__ import annotations

import argparse
import csv
import os
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# 守门链（顺序即 trainer.rs 里的优先级）
GATES = [
    ("gate_clinic", "守门·治病优先"),
    ("gate_camp", "守门·合宿全练"),
    ("gate_friend", "守门·友人外出截止"),
    ("gate_vital", "守门·体力危险"),
    ("gate_mood_friend", "守门·心情崩坏→友人出行"),
    ("gate_mood_outing", "守门·心情崩坏→普通出行"),
]

FUSION_RE = re.compile(r"【教诲】T(\d+) 融合 演目(\d+): (\d+)→(\d+)")
ASSIGN_RE = re.compile(r"【教诲】T(\d+) 分配决策 方案 (\S+)")
# 落地后的槽内容（base 教诲 id + 融合对象 id），= 「这次到底选了什么 buff」
ASSIGN_RESULT_RE = re.compile(r"【教诲】T(\d+) 分配 slots=\[([^\]]*)\] fused=\[([^\]]*)\]")
ACTLV_RE = re.compile(r"演目Lv \[([0-9,\s]+)\]（上限 (\d+)）")
SCORE_RE = re.compile(r"^评分\s+(\d+)\s+(\S+)")
PT_RE = re.compile(r"^PT\s+(\d+)")

COLUMNS = [
    "seed", "score", "rank",
    "pin_window", "pin_pos", "pin_ok",
    "speed", "stamina", "power", "guts", "wisdom", "five_sum", "pt",
    "act_levels", "act_level_max",
    "assign_count", "assign_picks",
    "fusion_count", "assist_gain", "assist_use",
    "kensan_hit", "kensan_used", "kensan_wasted",
    *[g[0] for g in GATES],
    "gate_total", "parse_ok",
]


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def five_status(stdout: str) -> list[int] | None:
    """从终局块的五维表取值：优先最后一个恰好含 5 个整数的 │ 行，退回任意 5 整数行。"""
    candidates = [ln for ln in stdout.splitlines() if "│" in ln]
    for ln in reversed(candidates):
        nums = [int(x) for x in re.findall(r"\d+", ln)]
        if len(nums) == 5:
            return nums
    for ln in reversed(stdout.splitlines()):
        nums = [int(x) for x in re.findall(r"\d+", ln)]
        if len(nums) == 5 and not ln.startswith(("回合", "阶段", "决策")):
            return nums
    return None


def parse_game(stdout: str, stderr: str) -> dict:
    out = strip_ansi(stdout)
    err = strip_ansi(stderr)

    row: dict = {c: "" for c in COLUMNS}

    for ln in out.splitlines():
        if m := SCORE_RE.match(ln.strip()):
            row["score"], row["rank"] = m.group(1), m.group(2)
        elif m := PT_RE.match(ln.strip()):
            row["pt"] = m.group(1)

    stats = five_status(out)
    if stats:
        for key, value in zip(("speed", "stamina", "power", "guts", "wisdom"), stats):
            row[key] = value
        row["five_sum"] = sum(stats)

    if m := ACTLV_RE.search(err):
        levels = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
        row["act_levels"] = "/".join(map(str, levels))
        row["act_level_max"] = m.group(2)

    picks = [f"T{t}:{scheme}" for t, scheme in ASSIGN_RE.findall(err)]
    row["assign_count"] = len(picks)
    row["assign_picks"] = "|".join(picks)

    row["pin_ok"] = int("钉位" in err)

    row["fusion_count"] = len(FUSION_RE.findall(err))
    row["assist_gain"] = len(re.findall(r"【助力支撑】T\d+ 稽古#\d+ 获得", err))
    row["assist_use"] = len(re.findall(r"【助力支撑】T\d+ 训练#\d+ 消耗", err))
    row["kensan_hit"] = len(re.findall(r"【追加稽古】T\d+ 演目\d+ 掷中", err))
    row["kensan_used"] = len(re.findall(r"【追加稽古】T\d+ 训练#\d+ 掷中 → 事件", err))
    row["kensan_wasted"] = len(re.findall(r"【追加稽古】T\d+ 掷中未兑现", err))

    total = 0
    for key, needle in GATES:
        n = err.count(needle)
        row[key] = n
        total += n
    row["gate_total"] = total

    row["parse_ok"] = int(bool(row["score"] and stats and row["act_levels"]))
    return row


def pin_summary(rows: list[dict]) -> dict:
    """钉位穷举汇总：每个互异落位的评分均值 / 局数 / 钉位是否真生效。"""
    groups: dict[int, list[int]] = {}
    for row in rows:
        if str(row.get("score", "")).isdigit():
            groups.setdefault(int(row["pin_pos"]), []).append(int(row["score"]))
    table = [
        {
            "pos": pos,
            "games": len(scores),
            "mean": round(sum(scores) / len(scores), 1),
            "min": min(scores),
            "max": max(scores),
        }
        for pos, scores in sorted(groups.items())
    ]
    best = max(table, key=lambda entry: entry["mean"]) if table else None
    return {
        "window": rows[0].get("pin_window") if rows else None,
        "distinct_positions": len(table),
        "pin_ok": sum(int(row.get("pin_ok") or 0) for row in rows),
        "best": best,
        "table": table,
    }


def assign_detail(stderr: str) -> list[str]:
    """buff 选择明细：每个教诲窗口选了哪个排列方案，落地后 5 槽各是什么。"""
    err = strip_ansi(stderr)
    picks = dict(ASSIGN_RE.findall(err))
    return [
        f"T{t} 方案 {picks.get(t, '?')} → slots[{slots.strip()}] fused[{fused.strip()}]"
        for t, slots, fused in ASSIGN_RESULT_RE.findall(err)
    ]


def fusion_detail(stderr: str) -> list[str]:
    """buff 替换明细：每次教诲融合 = 把 base 演目的教诲替换成 target。"""
    err = strip_ansi(stderr)
    return [f"T{t} 演目{a}: {b}→{c}" for t, a, b, c in FUSION_RE.findall(err)]


def run_one(
    binary: Path, seed: int, cards: str, repo: Path, env: dict[str, str] | None = None
) -> tuple[dict, str, str]:
    overlay = repo / "ai_config.toml"
    cards_toml = ", ".join(c.strip() for c in cards.split(",") if c.strip())
    overlay.write_text(f'[sim]\nseed = {seed}\ncards = [{cards_toml}]\n', encoding="utf8")
    proc = subprocess.run(
        [str(binary), "matsuri"],
        cwd=repo, capture_output=True, text=True, errors="replace", timeout=600,
        env={**os.environ, **env} if env else None,
    )
    row = parse_game(proc.stdout, proc.stderr)
    row["seed"] = seed
    # 终局块原文留档：解析式若失配，改解析器即可，不必重跑 100 局
    tail = "\n".join(strip_ansi(proc.stdout).splitlines()[-24:])
    return row, tail, strip_ansi(proc.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--cards", default="")
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--out", default="artifacts/matrix")
    ap.add_argument("--keep-logs", type=int, default=3, help="保留前 N 局完整原始输出")
    ap.add_argument("--pin-window", type=int, default=-1,
                    help="钉位穷举：教诲窗口下标 0-5（turn 2/11/23/35/47/59）；-1 = 不钉位")
    ap.add_argument("--pin-positions", type=int, default=0,
                    help="钉位穷举的互异落位数；0 = 自动（窗口 4/5 为 24，其余 120）")
    ap.add_argument("--pin-reps", type=int, default=1,
                    help="每个落位重复几个种子（CRN 配对；总数仍受 100 局硬上限约束")
    args = ap.parse_args()

    repo = Path.cwd()
    binary = (repo / args.binary).resolve()
    if not binary.exists():
        print(f"找不到可执行文件: {binary}", file=sys.stderr)
        return 2

    cards = args.cards.strip()
    if not cards:
        m = re.search(r"^cards\s*=\s*\[([^\]]+)\]",
                      (repo / "data/ai_config.toml").read_text(encoding="utf8"), re.M)
        if not m:
            print("data/ai_config.toml 里没有 cards", file=sys.stderr)
            return 2
        cards = m.group(1)
    print(f"卡组: {cards}")

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    if args.pin_window >= 0:
        positions = args.pin_positions or (24 if args.pin_window >= 4 else 120)
        plan = [(args.start_seed + rep, args.pin_window, pos)
                for pos in range(positions) for rep in range(args.pin_reps)]
    else:
        plan = [(args.start_seed + i, -1, -1) for i in range(args.count)]
    if len(plan) > 100:
        print(f"计划 {len(plan)} 局超过硬上限 100，截断", file=sys.stderr)
        plan = plan[:100]
    args.count = len(plan)

    rows, all_fusions, gate_counter = [], Counter(), Counter()
    assign_counter, assign_lines = Counter(), []
    with (out / "matrix.csv").open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i, (seed, window, pos) in enumerate(plan):
            env = {"MATSURI_ASSIGN_PIN": f"{window}:d{pos}"} if window >= 0 else None
            try:
                row, tail, err = run_one(binary, seed, cards, repo, env)
            except subprocess.TimeoutExpired:
                print(f"  seed {seed} 超时", file=sys.stderr)
                row, tail, err = {"seed": seed, "parse_ok": 0}, "TIMEOUT", ""
            row["pin_window"], row["pin_pos"] = window, pos
            rows.append(row)
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
            fh.flush()
            (out / "raw" / f"{seed}.tail.txt").write_text(tail, encoding="utf8")
            if i < args.keep_logs:
                (out / "raw" / f"{seed}.full.log").write_text(err, encoding="utf8")
            for d in fusion_detail(err):
                all_fusions[d.split(" ", 1)[1]] += 1
            details = assign_detail(err)
            if details:
                assign_lines.append(f"## seed {seed}\n" + "\n".join(details))
            for line in details:
                # 明细行形如「T2 方案 41320 → slots[..] fused[..]」，取第 3 段 = 方案本身
                parts = line.split(" ")
                if len(parts) > 2:
                    assign_counter[f"{parts[0]} {parts[2]}"] += 1
            for key, needle in GATES:
                if row.get(key):
                    gate_counter[key] += int(row[key])
            if (i + 1) % 10 == 0:
                print(f"  进度 {i + 1}/{args.count}", file=sys.stderr)

    (repo / "ai_config.toml").unlink(missing_ok=True)

    scored = [int(r["score"]) for r in rows if str(r.get("score", "")).isdigit()]
    summary = {
        "count": args.count,
        "seeds": f"{args.start_seed}-{args.start_seed + args.count - 1}",
        "cards": cards,
        "parsed": len(scored),
        "mean": round(sum(scored) / len(scored), 1) if scored else None,
        "min": min(scored) if scored else None,
        "max": max(scored) if scored else None,
        "gate_hits": dict(gate_counter),
        "fusion_top": all_fusions.most_common(20),
        "assign_top": assign_counter.most_common(20),
        "pin": pin_summary(rows) if args.pin_window >= 0 else None,
        "assign_windows": round(sum(assign_counter.values()) / max(len(rows), 1), 2),
    }
    (out / "assign_detail.md").write_text(
        "\n\n".join(assign_lines), encoding="utf8")
    for axis in ("speed", "stamina", "power", "guts", "wisdom", "five_sum", "pt"):
        vals = [int(r[axis]) for r in rows if str(r.get(axis, "")).lstrip("-").isdigit()]
        summary[f"mean_{axis}"] = round(sum(vals) / len(vals), 1) if vals else None
    ranks = Counter(str(r.get("rank", "?"))[:2] for r in rows if r.get("rank"))
    summary["rank_prefix"] = dict(ranks.most_common())

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["parsed"] == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
