"""把一局游戏的原始输出整理成**逐回合 CSV**（局内回合数据）。

模拟器一局会打两份东西：
- stdout：每回合一行状态快照 `T60  体 26/104 干劲 绝好调 速 2458 耐 1523 力 1271 根 1042 智 1799 PT 6637`
  + 终局块（评分 / 评级 / PT / 五维分项表格）
- stderr：`[INFO]` 诊断流（【回合】【事件】【决策】【动作】【教诲】【分布】【搜索】【终局】）

这个脚本按回合号把两边对齐，输出一行一回合的 CSV：这一回合是哪种回合、AI 选了哪个动作、
打分是多少、教诲窗口的方案与落位（slots/fused）、五演目人头分布、动作结算的属性/体力/PT 变化，
以及（开了蒙特卡洛时）搜索的候选数、rollout 次数、中选位次和搜索值。

用途：复盘「为什么这一局低分」不用再翻 1800 行日志；也是把局内轨迹导给外部分析
（例如攒神经网络的教师数据、做符号回归的样本表）的入口。

用法：
    python3 scripts/turns_to_csv.py --stdout <seed>.stdout.txt --stderr <seed>.stderr.txt \
        --seed 62100 --out turns/62100.csv
    python3 scripts/turns_to_csv.py --dir logs/champion-ga-r2 --out-dir turns/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# stdout：逐回合状态快照
SNAP_RE = re.compile(
    r"^T(?P<turn>\d+)\s+体 (?P<vital>-?\d+)/(?P<vital_max>\d+)\s+干劲 (?P<motivation>\S+)\s+"
    r"速 (?P<speed>\d+) 耐 (?P<stamina>\d+) 力 (?P<power>\d+) 根 (?P<guts>\d+) "
    r"智 (?P<wisdom>\d+) PT (?P<pt>\d+)\s*$")
# stdout：终局块
SCORE_RE = re.compile(r"^评分\s+(\d+)\s+(\S+)", re.M)
PT_FINAL_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")

# stderr：诊断流
TURN_TYPE_RE = re.compile(r"【回合】T(\d+)\s+(.+?)\s+类型:\s*(.*)$")
ACTION_RE = re.compile(r"【决策】T(\d+)\s+动作\s+(\S+)\s+值=([\d.]+)")
EVENT_RE = re.compile(r"【决策】T(\d+)\s+(事件#\S+\s+\S+\s+选项\d+)\s+期望=([\d.-]+)")
SETTLE_RE = re.compile(r"【动作】T(\d+)\s+(\S+)\s+(成功|失败)（失败率 ([\d.]+)%）｜变化\s*(.*)$")
DIST_RE = re.compile(r"【分布】T(\d+)\s+人数 \[([0-9, ]+)\](?:\s+闪\[([0-9, ]+)\])?")
ADVICE_PLAN_RE = re.compile(r"【决策】T(\d+)\s+教诲分配\s+值=([\d.]+)\s+方案\s+(\S+)")
ADVICE_SLOTS_RE = re.compile(r"【教诲】T(\d+)\s+分配 slots=\[([0-9, ]+)\] fused=\[([0-9, ]+)\]")
ADVICE_FUSION_RE = re.compile(r"【教诲】T(\d+)\s+融合 演目(\d+): (\d+)→(\d+)")
SEARCH_RE = re.compile(r"【搜索】T(\d+)\s+教诲 MC 候选 (\d+) 个 × (\d+) 次 → 中选第 (\d+) 位 值=([\d.]+)")
ACT_LV_RE = re.compile(r"演目Lv \[([0-9, ]+)\]")
RACE_RE = re.compile(r"【终局】T(\d+)\s+三场结果 \[([^\]]+)\]")

COLUMNS = [
    "seed", "turn", "phase", "turn_type", "vital", "vital_max", "motivation",
    "speed", "stamina", "power", "guts", "wisdom", "pt",
    "action", "action_value", "event_choice", "event_expect",
    "settle", "settle_delta", "fail_rate",
    "persons", "shining",
    "advice_plan", "advice_value", "advice_slots", "advice_fused", "advice_fusion",
    "search_cands", "search_n", "search_pick", "search_value",
]


def blank(seed: str, turn: int) -> dict:
    row = {c: "" for c in COLUMNS}
    row["seed"], row["turn"] = seed, turn
    return row


def parse(stdout: str, stderr: str, seed: str) -> tuple[list[dict], dict]:
    rows: dict[int, dict] = {}

    def at(turn: int) -> dict:
        return rows.setdefault(turn, blank(seed, turn))

    for line in stdout.splitlines():
        m = SNAP_RE.match(line.strip())
        if m:
            row = at(int(m["turn"]))
            for key in ("vital", "vital_max", "motivation", "speed", "stamina",
                        "power", "guts", "wisdom", "pt"):
                row[key] = m[key]

    for raw in stderr.splitlines():
        line = raw[raw.find("]") + 1:].strip() if raw.startswith("[") else raw.strip()
        for regex, handler in (
            (TURN_TYPE_RE, lambda m, r: r.update(phase=m.group(2), turn_type=m.group(3))),
            (ACTION_RE, lambda m, r: r.update(action=m.group(2), action_value=m.group(3))),
            (EVENT_RE, lambda m, r: r.update(event_choice=m.group(2), event_expect=m.group(3))),
            (SETTLE_RE, lambda m, r: r.update(settle=f"{m.group(2)} {m.group(3)}",
                                              fail_rate=m.group(4), settle_delta=m.group(5))),
            (DIST_RE, lambda m, r: r.update(persons=m.group(2).strip(),
                                            shining=(m.group(3) or "").strip())),
            (ADVICE_PLAN_RE, lambda m, r: r.update(advice_value=m.group(2),
                                                   advice_plan=m.group(3))),
            (ADVICE_SLOTS_RE, lambda m, r: r.update(advice_slots=m.group(2).strip(),
                                                    advice_fused=m.group(3).strip())),
            (ADVICE_FUSION_RE, lambda m, r: r.__setitem__(
                "advice_fusion",
                (r["advice_fusion"] + ";" if r["advice_fusion"] else "")
                + f"演目{m.group(2)}:{m.group(3)}→{m.group(4)}")),
            (SEARCH_RE, lambda m, r: r.update(search_cands=m.group(2), search_n=m.group(3),
                                              search_pick=m.group(4), search_value=m.group(5))),
        ):
            m = regex.search(line)
            if m:
                handler(m, at(int(m.group(1))))
                break

    final = {"seed": seed}
    m = SCORE_RE.search(stdout)
    if m:
        final["score"], final["rank"] = int(m.group(1)), m.group(2)
    m = PT_FINAL_RE.search(stdout)
    if m:
        final["pt"] = int(m.group(1))
    m = FIVE_RE.search(stdout)
    if m:
        for key, value in zip(("speed", "stamina", "power", "guts", "wisdom"), m.groups()):
            final[key] = int(value)
        final["five_sum"] = sum(int(x) for x in m.groups())
    m = ACT_LV_RE.search(stderr)
    if m:
        final["act_lv"] = [int(x) for x in m.group(1).split(",")]
    m = RACE_RE.search(stderr)
    if m:
        final["race_result"] = m.group(2)
    return [rows[k] for k in sorted(rows)], final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", help="单局：stdout 文件")
    ap.add_argument("--stderr", help="单局：stderr 文件")
    ap.add_argument("--seed", default="", help="单局：种子号（写进 CSV 第一列）")
    ap.add_argument("--out", default="", help="单局：输出 CSV 路径")
    ap.add_argument("--dir", default="", help="批量：含 <seed>.stdout.txt/<seed>.stderr.txt 的目录")
    ap.add_argument("--out-dir", default="", help="批量：CSV 输出目录")
    args = ap.parse_args()

    jobs: list[tuple[str, Path, Path]] = []
    if args.dir:
        for so in sorted(Path(args.dir).glob("*.stdout.txt")):
            se = so.with_name(so.name.replace(".stdout.txt", ".stderr.txt"))
            if se.exists():
                jobs.append((so.stem.split(".")[0], so, se))
    elif args.stdout and args.stderr:
        seed = args.seed or Path(args.stdout).stem.split(".")[0]
        jobs.append((seed, Path(args.stdout), Path(args.stderr)))
    else:
        print("需要 --dir，或者 --stdout + --stderr", file=sys.stderr)
        return 2

    if not jobs:
        print("没找到成对的 stdout/stderr 文件", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.out).parent if args.out
                                                       else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)
    finals = []
    for seed, so, se in jobs:
        rows, final = parse(so.read_text(encoding="utf8", errors="replace"),
                            se.read_text(encoding="utf8", errors="replace"), seed)
        target = Path(args.out) if (args.out and len(jobs) == 1) else out_dir / f"{seed}.turns.csv"
        with target.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        finals.append(final)
        print(f"{seed}: {len(rows)} 个回合 → {target}"
              f"（评分 {final.get('score', '?')} {final.get('rank', '')}"
              f" 演目Lv {final.get('act_lv', '')}）")
    (out_dir / "finals.json").write_text(json.dumps(finals, ensure_ascii=False, indent=2),
                                        encoding="utf8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
