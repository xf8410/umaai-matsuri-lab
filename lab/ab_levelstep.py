#!/usr/bin/env python3
"""攻略优化点 4（吾妻外出 Lv8/9 质变卡阶）· CRN A/B 验证

同一批种子跑 3 臂：level_step_bonus = 0（对照）/ 120 / 250。
每臂 ≤100 局（默认 50），CRN 配对 → 配对 t 检验。
采集：育成评分 / 五维分列 + 合计 / PT / 演目Lv（速耐力根智）/ 友人出行次数 / 稽古次数。

注入路径：直接改 data/matsuri_config.toml 的 level_step_bonus（[trainer] 覆盖层是整段替换，
逐字段 overlay 会丢旋钮，所以走「改默认层 + try/finally 还原」）。sim 层（seed/cards/uma）
仍走仓根 ai_config.toml overlay。跑完还原两文件，工作树保持干净。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from matrix import strip_ansi, five_status  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "target/release/matsuri"
CFG = REPO / "data/matsuri_config.toml"
OVERLAY = REPO / "ai_config.toml"

ACT = ["速", "耐", "力", "根", "智"]
CARD_ACTS = [0, 1, 2, 4]  # 根性(act3)无卡，质变奖励自动排除
SCORE_RE = re.compile(r"^评分\s+(\d+)\s+(\S+)")
PT_RE = re.compile(r"^PT\s+(\d+)")
ACTLV_RE = re.compile(r"演目Lv \[([0-9,\s]+)\]（上限 (\d+)）")
DECIDE_RE = re.compile(r"【决策】T\d+ 动作 (\S+)")


def set_bonus(value: float) -> None:
    text = CFG.read_text(encoding="utf8")
    new, n = re.subn(r"^level_step_bonus = .*$",
                     f"level_step_bonus = {value}", text, flags=re.M)
    if n != 1:
        raise RuntimeError(f"level_step_bonus 替换命中 {n} 次（应为 1）")
    CFG.write_text(new, encoding="utf8")


def run_one(seed: int, cards: str, uma: str) -> dict:
    OVERLAY.write_text(
        f"[sim]\nseed = {seed}\ncards = [{cards}]\numa = {uma}\n", encoding="utf8")
    p = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                       text=True, errors="replace", timeout=600)
    out, err = strip_ansi(p.stdout), strip_ansi(p.stderr)
    row: dict = {"seed": seed, "ok": 0}
    for ln in out.splitlines():
        if m := SCORE_RE.match(ln.strip()):
            row["score"], row["rank"] = int(m.group(1)), m.group(2)
        elif m := PT_RE.match(ln.strip()):
            row["pt"] = int(m.group(1))
    st = five_status(out)
    if st:
        for k, v in zip(("speed", "stamina", "power", "guts", "wisdom"), st):
            row[k] = v
        row["five_sum"] = sum(st)
    if m := ACTLV_RE.search(err):
        lv = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
        row["act_levels"] = lv
        row["lv8"] = sum(lv[a] >= 8 for a in CARD_ACTS)
        row["lv9"] = sum(lv[a] >= 9 for a in CARD_ACTS)
    acts = DECIDE_RE.findall(err)
    row["outing"] = sum(a == "友人出行" for a in acts)
    row["keiko"] = sum(a.startswith("稽古") for a in acts)
    row["ok"] = int("score" in row and st is not None and "act_levels" in row)
    return row


def paired(ctrl: list[float], treat: list[float], name: str) -> str:
    c, t = np.array(ctrl, float), np.array(treat, float)
    d = t - c
    tstat, p = stats.ttest_rel(t, c)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return (f"  {name:<10} 对照 {c.mean():8.1f} → 处理 {t.mean():8.1f}  "
            f"Δ{d.mean():+7.1f} ± {d.std(ddof=1)/np.sqrt(len(d)):5.1f}  "
            f"t={tstat:+6.2f} p={p:.4f} {star}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--arms", default="0,120,250")
    ap.add_argument("--cards", default="303174, 303114, 303004, 303224, 303204, 303234")
    ap.add_argument("--uma", default="100101")
    ap.add_argument("--out", default="lab/ab_levelstep")
    args = ap.parse_args()

    if int(args.uma) in (106301, 106302):
        print("禁用马，拒跑", file=sys.stderr); return 2
    arms = [float(x) for x in args.arms.split(",")]
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    if len(seeds) > 100:
        print("每臂 >100 局，违反硬上限", file=sys.stderr); return 2

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    backup = CFG.read_text(encoding="utf8")
    results: dict[float, list[dict]] = {}
    try:
        for bonus in arms:
            set_bonus(bonus)
            print(f"\n=== 臂 level_step_bonus={bonus:g} · {len(seeds)} 局 "
                  f"(seeds {seeds[0]}-{seeds[-1]}) ===", file=sys.stderr)
            rows = []
            for i, s in enumerate(seeds):
                rows.append(run_one(s, args.cards, args.uma))
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(seeds)}", file=sys.stderr)
            results[bonus] = rows
            ok = [r for r in rows if r["ok"]]
            sc = np.array([r["score"] for r in ok], float)
            print(f"  解析 {len(ok)}/{len(rows)}  评分 mean {sc.mean():.0f} "
                  f"[{sc.min():.0f},{sc.max():.0f}]", file=sys.stderr)
    finally:
        CFG.write_text(backup, encoding="utf8")
        OVERLAY.unlink(missing_ok=True)
        print("\n[还原] data/matsuri_config.toml + 删除 ai_config.toml overlay",
              file=sys.stderr)

    # 报告
    ctrl = arms[0]
    cc = [r for r in results[ctrl] if r["ok"]]
    lines = ["# 攻略点4 质变卡阶 level_step_bonus · CRN A/B\n",
             f"对照臂 = {ctrl:g}；每臂 {len(cc)} 局，seeds {seeds[0]}-{seeds[-1]}；"
             f"uma={args.uma}；卡组={args.cards}\n"]

    def col(rows, k):
        return [r[k] for r in rows if k in r]

    # 对照臂基线
    lines.append(f"\n## 对照臂 level_step_bonus={ctrl:g} 基线")
    for k, lbl in [("score", "育成评分"), ("five_sum", "五维合计"), ("pt", "PT"),
                   ("lv8", "演目Lv≥8(有卡4项)"), ("lv9", "演目Lv≥9(有卡4项)"),
                   ("outing", "友人出行次数"), ("keiko", "稽古次数")]:
        v = np.array(col(cc, k), float)
        lines.append(f"- {lbl}: {v.mean():.1f} ± {v.std(ddof=1):.1f}")
    for j, a in enumerate(ACT):
        v = np.array([r["act_levels"][j] for r in cc if "act_levels" in r], float)
        lines.append(f"- 演目Lv {a}: {v.mean():.2f}")
    for j, a in enumerate(("速", "耐", "力", "根", "智")):
        v = np.array(col(cc, ("speed", "stamina", "power", "guts", "wisdom")[j]), float)
        lines.append(f"- 属性 {a}: {v.mean():.0f}")

    # 各处理臂 vs 对照（CRN 配对，按 seed 对齐）
    for bonus in arms[1:]:
        tt = [r for r in results[bonus] if r["ok"]]
        idx = {r["seed"]: r for r in tt}
        pairs = [(c, idx[c["seed"]]) for c in cc if c["seed"] in idx]
        lines.append(f"\n## 处理臂 level_step_bonus={bonus:g} vs 对照（配对 n={len(pairs)}）")
        for k, lbl in [("score", "育成评分"), ("five_sum", "五维合计"), ("pt", "PT"),
                       ("speed", "速"), ("stamina", "耐"), ("power", "力"),
                       ("guts", "根"), ("wisdom", "智"),
                       ("lv8", "Lv≥8数"), ("lv9", "Lv≥9数"),
                       ("outing", "友人出行"), ("keiko", "稽古")]:
            lines.append(paired([c[k] for c, _ in pairs], [t[k] for _, t in pairs], lbl))

    report = "\n".join(lines)
    (out / "REPORT.md").write_text(report + "\n", encoding="utf8")
    import json
    (out / "raw.json").write_text(
        json.dumps({str(k): v for k, v in results.items()}, ensure_ascii=False),
        encoding="utf8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
