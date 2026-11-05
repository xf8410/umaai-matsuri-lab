"""教示 buff 路线的坐标上升穷举（骏大祭）。

目标：给定一套固定配卡 + 固定育成马，逐窗口穷举「哪个 buff 放哪个演目」，
搜出一条比攻略路线 / AI 自选都高分的教示路线。

为什么不是一把梭全穷举：6 个窗口 × 120 落位 = 120^6，不可能。用**坐标上升**——
一次只放开一个窗口，其余窗口钉在当前最优路线上，逐窗口收敛。这是决策剪枝的常规做法，
也是玩家实际的思路（先定序盘，再定合宿，最后定收尾）。

三阶段，每阶段的游戏数都压在仓库的 100 局硬上限内：
  1. 筛选：该窗口全部互异落位 × 3 个种子（窗口 0–3 = 120 落位 → 360 局，窗口 4/5 = 24 落位 → 72 局）。
     这一步是**单种子级扫描**，只用来砍掉明显的坏落位，不做统计结论。
  2. 确认：筛选前 10 名 × 10 个种子 = 100 局（正好一个矩阵），按均分选窗口最优。
  3. 验收：整条路线 vs 攻略路线，在**留出种子**（搜索全程没用过）上跑 100 局配对 t。

钉位用语义写法（buff id）而不是 `d<N>`：`d<N>` 的下标依赖当时的 `(slots, fused)` 状态，
跨窗口不直观；id 串（`<id0>-<id1>-<id2>-<id3>-<id4>`）在 `apply_assignment` 落地后匹配，
写进 RESULTS 也能直接被人读懂。筛选阶段临时用 `d<N>`，选中后从日志反解成 id 串。

跑局用一个独立的临时工作目录（`data` 软链到仓库 `data/`），所以可以和 GA 并行，
不会跟 GA 抢仓库根的 `matsuri_config.toml` / `ai_config.toml`。

用法：
    python3 scripts/route_search.py --cards 302824,303114,303074,303224,302894,303234 \
        --uma 102102 --overlay lab/config/d-status.toml --out artifacts/route-S
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")
# 「【教诲】T11 分配 slots=[501, 201, 301, 101, 401] fused=[0, 0, 0, 0, 0]」
SLOTS_RE = re.compile(r"分配 slots=\[([0-9, ]+)\] fused=\[([0-9, ]+)\]")
PIN_RE = re.compile(r"钉位 窗口(\d+)")
ACT_NAMES = ["速", "耐", "力", "根", "智"]


def make_workdir(repo: Path, overlay: Path) -> Path:
    """独立工作目录：data 软链 + 覆盖层拷贝，避免与并行的 GA 抢仓库根配置。"""
    work = Path(f"/tmp/route-search-{overlay.stem}")
    work.mkdir(parents=True, exist_ok=True)
    link = work / "data"
    if not link.exists():
        link.symlink_to(repo / "data")
    (work / "matsuri_config.toml").write_text(overlay.read_text(encoding="utf8"),
                                              encoding="utf8")
    return work


def play(binary: Path, work: Path, seed: int, uma: str, cards: str, pin: str) -> dict:
    """跑一局，返回 {score, pt, five, lv, slots}；score=0 表示没跑出终局块。"""
    cards_toml = ", ".join(c.strip() for c in cards.split(",") if c.strip())
    (work / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {uma}\ncards = [{cards_toml}]\n", encoding="utf8")
    env = {**os.environ, "MATSURI_ASSIGN_PIN": pin} if pin else None
    proc = subprocess.run([str(binary), "matsuri"], cwd=work, capture_output=True,
                          text=True, errors="replace", timeout=600, env=env)
    out, err = proc.stdout, proc.stderr
    m = SCORE_RE.search(out)
    if not m:
        return {"score": 0, "err": (out + err)[-300:]}
    five = FIVE_RE.search(out)
    lv = LV_RE.search(err)
    pt = PT_RE.search(out)
    return {
        "score": int(m.group(1)),
        "pt": int(pt.group(1)) if pt else 0,
        "five": [int(x) for x in five.groups()] if five else [],
        "lv": [int(x) for x in lv.group(1).split(",")] if lv else [],
        "slots": [(list(map(int, a.split(","))), list(map(int, b.split(","))))
                  for a, b in SLOTS_RE.findall(err)],
        "pin_windows": sorted({int(x) for x in PIN_RE.findall(err)}),
    }


def route_with(route: dict[int, str], window: int, value: str) -> str:
    """把某个窗口换成 `value`，其余窗口保持当前最优，拼成 MATSURI_ASSIGN_PIN。"""
    merged = dict(route)
    merged[window] = value
    return ",".join(f"{w}:{merged[w]}" for w in sorted(merged))


def distinct_count(binary: Path, work: Path, args, route: dict[int, str],
                   window: int) -> int:
    """该窗口的互异落位数：`d<N>` 逐个往上探，第一次没命中钉位就是边界。"""
    n = 0
    while n < 130:
        r = play(binary, work, args.start_seed, args.uma, args.cards,
                 route_with(route, window, f"d{n}"))
        if window not in r.get("pin_windows", ()):
            break
        n += 1
    return n


def ids_from_slots(slots: list[int], fused: list[int], prefer_fused: bool = True) -> str:
    """把落地后的 (slots, fused) 反解成语义 id 串：有転成写転成 id，否则写基础 id。"""
    out = []
    for act in range(5):
        value = fused[act] if (prefer_fused and fused[act]) else slots[act]
        out.append(str(value or 0))
    return "-".join(out)


def paired(a: list[int], b: list[int]) -> tuple[float, float, int]:
    """配对差 (均值, t, a 胜局数)。CRN 同种子，种子主效应被消掉。"""
    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    mean = sum(diff) / n
    sd = st.pstdev(diff)
    t = mean / (sd / (n ** 0.5)) if sd else 0.0
    return mean, t, sum(1 for x in diff if x > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True)
    ap.add_argument("--uma", default="102102")
    ap.add_argument("--overlay", default="lab/config/d-status.toml")
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="artifacts/route-search")
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--screen-seeds", type=int, default=3, help="筛选阶段每个落位跑几个种子")
    ap.add_argument("--topk", type=int, default=10, help="进入确认阶段的落位数")
    ap.add_argument("--confirm-seeds", type=int, default=10, help="确认阶段每个落位跑几个种子")
    ap.add_argument("--windows", default="0,1,2,3,4,5")
    ap.add_argument("--base-route", default="",
                    help="起点路线；留空 = 全部窗口交给 AI 自选")
    ap.add_argument("--guide-route",
                    default="0:101-201-301-501-401,1:501-201-301-101-401,"
                            "2:601-501-301-401-201,3:601-301-501-401-201,"
                            "4:801-301-0-401-201,5:801-301-0-401-201")
    ap.add_argument("--holdout-seed", type=int, default=61544,
                    help="验收种子起点（搜索全程不用，避免自证）")
    ap.add_argument("--holdout-count", type=int, default=100)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    binary = (repo / args.binary).resolve()
    if not binary.exists():
        print(f"找不到可执行文件: {binary}", file=sys.stderr)
        return 2
    overlay = (repo / args.overlay).resolve()
    work = make_workdir(repo, overlay)
    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"工作目录 {work}（data 软链，不碰仓库根配置）", flush=True)

    def parse_route(text: str) -> dict[int, str]:
        route = {}
        for entry in text.split(","):
            entry = entry.strip()
            if not entry:
                continue
            w, _, value = entry.partition(":")
            route[int(w)] = value
        return route

    route = parse_route(args.base_route)
    t0 = time.time()
    log: list[dict] = []

    # 参考基线：攻略路线 & AI 自选，都在验收种子上跑，后面所有对比都对齐这套种子
    holdout = [args.holdout_seed + i for i in range(args.holdout_count)]

    for window in [int(w) for w in args.windows.split(",")]:
        count = distinct_count(binary, work, args, route, window)
        if count == 0:
            print(f"窗口 {window}: 探不到任何互异落位，跳过", flush=True)
            continue
        print(f"\n=== 窗口 {window}（turn {[2, 11, 23, 35, 47, 59][window]}）"
              f"互异落位 {count} 个 ===", flush=True)

        # 阶段 1 筛选：全落位 × screen_seeds 个种子（单种子级扫描，只砍明显坏落位）
        screen_seeds = [args.start_seed + i for i in range(args.screen_seeds)]
        scored: list[tuple[float, int, str]] = []
        for n in range(count):
            pin = route_with(route, window, f"d{n}")
            runs = [play(binary, work, s, args.uma, args.cards, pin) for s in screen_seeds]
            scores = [r["score"] for r in runs if r["score"]]
            if not scores:
                continue
            # 反解语义 id 串：直接用第一个种子的落地结果，不额外跑局
            ids = ""
            first = runs[0]
            if len(first.get("slots", [])) > window:
                slots, fused = first["slots"][window]
                ids = ids_from_slots(slots, fused)
            scored.append((sum(scores) / len(scores), n, ids))
        scored.sort(reverse=True)
        top = scored[:args.topk]
        print(f"  筛选 {len(scored)} 个落位 × {len(screen_seeds)} 种子 "
              f"({time.time() - t0:.0f}s)，前 {len(top)}: "
              + ", ".join(f"d{n}={m:.0f}" for m, n, _ in top), flush=True)

        # 阶段 2 确认：前 K × confirm_seeds（≤100 局 = 一个矩阵）
        seeds = [args.start_seed + i for i in range(args.confirm_seeds)]
        best = None
        rows = []
        for mean_screen, n, ids in top:
            pin = route_with(route, window, f"d{n}")
            scores = [play(binary, work, s, args.uma, args.cards, pin)["score"]
                      for s in seeds]
            scores = [s for s in scores if s]
            if not scores:
                continue
            m = sum(scores) / len(scores)
            rows.append({"d": n, "ids": ids, "screen": round(mean_screen, 1),
                         "confirm": round(m, 1), "n": len(scores),
                         "min": min(scores), "max": max(scores)})
            if best is None or m > best[0]:
                best = (m, n, ids)
        rows.sort(key=lambda r: -r["confirm"])
        for r in rows:
            print(f"    d{r['d']:<4} {r['ids'] or '(未反解)':<26} "
                  f"筛选 {r['screen']:>8.1f} → 确认 {r['confirm']:>8.1f} "
                  f"(n={r['n']}, {r['min']}–{r['max']})", flush=True)
        if best is None:
            continue
        value = best[2] or f"d{best[1]}"
        # id 串反解失败就退回 d<N>（下标依赖状态，但至少能钉住）
        route[window] = value
        entry = {"window": window, "turn": [2, 11, 23, 35, 47, 59][window],
                 "distinct": count, "chosen": value, "confirm_mean": round(best[0], 1),
                 "candidates": rows}
        log.append(entry)
        (out / "search.json").write_text(json.dumps(
            {"route": route, "log": log, "elapsed_sec": round(time.time() - t0)},
            ensure_ascii=False, indent=2), encoding="utf8")
        print(f"  → 窗口 {window} 定为 {value}（确认均分 {best[0]:.1f}）", flush=True)

    final = ",".join(f"{w}:{route[w]}" for w in sorted(route))
    print(f"\n搜出的路线: {final}", flush=True)

    # 阶段 3 验收：留出种子（搜索没用过）上三方对比 —— 搜出路线 / 攻略路线 / AI 自选
    def run_route(pin: str) -> list[dict]:
        return [play(binary, work, s, args.uma, args.cards, pin) for s in holdout]

    print("验收：搜出路线 …", flush=True)
    found = run_route(final)
    print("验收：攻略路线 …", flush=True)
    gd = run_route(args.guide_route)
    print("验收：AI 自选 …", flush=True)
    ai = run_route("")

    def summarize(rs: list[dict]) -> dict:
        ok = [r for r in rs if r["score"]]
        n = len(ok)
        fives = [sum(r["five"]) for r in ok if r["five"]]
        lvs = [r["lv"] for r in ok if r["lv"]]
        return {
            "n": n,
            "mean": round(st.mean(r["score"] for r in ok), 1),
            "min": min(r["score"] for r in ok), "max": max(r["score"] for r in ok),
            "pt": round(st.mean(r["pt"] for r in ok), 1),
            "five": round(st.mean(fives), 1) if fives else None,
            "axes": [round(st.mean(r["five"][i] for r in ok if r["five"]), 1)
                     for i in range(5)] if fives else None,
            "act_lv": [round(st.mean(x[i] for x in lvs), 2) for i in range(5)]
                      if lvs else None,
            "ge_us1": sum(1 for r in ok if r["score"] >= 64200),
            "ge_us9": sum(1 for r in ok if r["score"] >= 74400),
            "ge_lg": sum(1 for r in ok if r["score"] >= 76000),
            "scores": [r["score"] for r in ok],
        }

    report = {"route_found": final, "route_guide": args.guide_route,
              "holdout_seeds": f"{holdout[0]}-{holdout[-1]}",
              "found": summarize(found), "guide": summarize(gd), "ai": summarize(ai)}
    a = [r["score"] for r in found if r["score"]]
    for name, other in (("found_vs_guide", gd), ("found_vs_ai", ai)):
        b = [r["score"] for r in other if r["score"]]
        m, tval, w = paired(a, b)
        report[name] = {"delta": round(m, 1), "t": round(tval, 2), "wins": w, "n": len(b)}
    (out / "holdout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf8")
    slim = {k: ({kk: vv for kk, vv in v.items() if kk != "scores"}
                if isinstance(v, dict) else v) for k, v in report.items()}
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
