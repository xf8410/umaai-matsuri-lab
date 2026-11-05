#!/usr/bin/env python3
"""教诲路线的禁忌搜索（离散决策优化）

坐标上升只会「每个窗口取当前最好」，一旦某窗口的好落位依赖别的窗口的选择就出不来
（局部最优）。禁忌搜索允许**接受变差的移动**来翻过山头，靠禁忌表防止原地打转：

- 邻域：选一个窗口，把它的落位换成任意另一个互异落位（其余窗口不动）
- 禁忌：被撤销的那个 `(窗口, 落位)` 进禁忌表，`--tenure` 轮内不许再选
- 渴望：即使被禁忌，只要它比**全局最优**还好，照样接受（否则可能错过最优解）
- 评价：邻域全扫用 `--screen-seeds` 个种子排序，中选移动再用 `--confirm-seeds`
  个种子确认（确认分才是记账口径）；同一条路线的评价走缓存，不重复跑局
- 验收：留出种子（搜索全程没用过）上三方对比 —— 禁忌路线 / 坐标上升路线 / 攻略路线

用法：
    python scripts/tabu_route.py --cards <卡组> --base-route <坐标上升冠军路线> \
        --overlay lab/config/ga-best.toml --passes 3 --tenure 3
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from route_search import (  # noqa: E402
    distinct_count,
    ids_from_slots,
    make_workdir,
    paired,
    play,
    route_with,
)

TURNS = [2, 11, 23, 35, 47, 59]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True)
    ap.add_argument("--uma", default="102102")
    ap.add_argument("--overlay", default="lab/config/ga-best.toml")
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="artifacts/tabu-route")
    ap.add_argument("--base-route", required=True, help="起点路线（坐标上升冠军）")
    ap.add_argument("--guide-route", default="", help="攻略路线（验收对照）")
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--screen-seeds", type=int, default=3)
    ap.add_argument("--confirm-seeds", type=int, default=10)
    ap.add_argument("--passes", type=int, default=3, help="轮数：每轮把 6 个窗口各扫一遍")
    ap.add_argument("--tenure", type=int, default=3, help="禁忌期（轮）")
    ap.add_argument("--windows", default="0,1,2,3,4,5")
    ap.add_argument("--holdout-seed", type=int, default=61900)
    ap.add_argument("--holdout-count", type=int, default=100)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    binary = (repo / args.binary).resolve()
    overlay = Path(args.overlay)
    if not binary.exists():
        print(f"找不到可执行文件: {binary}", file=sys.stderr)
        return 2
    if not overlay.exists():
        print(f"找不到覆盖层: {overlay}", file=sys.stderr)
        return 2

    work = make_workdir(repo, overlay)
    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"工作目录 {work} · 起点路线 {args.base_route}", flush=True)

    def parse_route(text: str) -> dict[int, str]:
        route: dict[int, str] = {}
        for entry in text.split(","):
            entry = entry.strip()
            if not entry:
                continue
            w, _, value = entry.partition(":")
            route[int(w)] = value
        return route

    def pin_of(route: dict[int, str]) -> str:
        return ",".join(f"{w}:{route[w]}" for w in sorted(route))

    screen = [args.start_seed + i for i in range(args.screen_seeds)]
    confirm = [args.start_seed + i for i in range(args.confirm_seeds)]
    cache: dict[str, dict[str, float]] = {}

    def evaluate(route: dict[int, str], seeds: list[int]) -> tuple[float, list[int]]:
        """按种子集评一条路线；同 (路线, 种子集) 走缓存。"""
        pin = pin_of(route)
        key = f"{pin}|{len(seeds)}"
        hit = cache.get(key)
        if hit is not None:
            return hit["mean"], hit["scores"]
        scores = [play(binary, work, s, args.uma, args.cards, pin)["score"] for s in seeds]
        scores = [s for s in scores if s]
        mean = st.mean(scores) if scores else 0.0
        cache[key] = {"mean": mean, "scores": scores}
        return mean, scores

    route = parse_route(args.base_route)
    cur_mean, _ = evaluate(route, confirm)
    best_route, best_mean = dict(route), cur_mean
    # 禁忌表：{(窗口, 落位): 解禁轮次}
    tabu: dict[tuple[int, str], int] = {}
    log: list[dict] = []
    windows = [int(w) for w in args.windows.split(",")]

    for it in range(1, args.passes + 1):
        print(f"\n===== 第 {it}/{args.passes} 轮（当前 {cur_mean:.1f}，"
              f"全局最优 {best_mean:.1f}，{time.time() - t0:.0f}s）=====", flush=True)
        for window in windows:
            count = distinct_count(binary, work, args, route, window)
            if count < 2:
                continue
            # 邻域全扫：先用少量种子排序，只给前几名做确认（省下的时间换更多轮）
            scored: list[tuple[float, int]] = []
            for n in range(count):
                cand = dict(route)
                cand[window] = f"d{n}"
                m, _ = evaluate(cand, screen)
                scored.append((m, n))
            scored.sort(reverse=True)
            picked: tuple[float, str, bool] | None = None
            for _, n in scored[:6]:
                value = f"d{n}"
                cand = dict(route)
                cand[window] = value
                banned = tabu.get((window, value), 0) >= it
                m, _ = evaluate(cand, confirm)
                # 渴望准则：禁忌中的落位只要打赢全局最优就照样接受
                if banned and m <= best_mean:
                    continue
                if picked is None or m > picked[0]:
                    picked = (m, value, banned)
            if picked is None:
                print(f"  窗口 {window}（T{TURNS[window]}）：全部邻域都在禁忌期，保持不动",
                      flush=True)
                continue
            mean, value, was_tabu = picked
            # 反解语义 id 串（有転成写転成 id）：落盘 / 复现都用它，不依赖状态相关的 d<N>
            probe = play(binary, work, screen[0], args.uma, args.cards,
                         route_with(route, window, value))
            ids = ""
            if len(probe.get("slots", [])) > window:
                slots, fused = probe["slots"][window]
                ids = ids_from_slots(slots, fused)
            # 被换掉的落位进禁忌表：tenure 轮内不许马上换回来，逼搜索往前走
            old = route[window]
            tabu[(window, old)] = it + args.tenure
            route[window] = ids or value
            cur_mean = mean
            accepted = mean > best_mean
            if accepted:
                best_route, best_mean = dict(route), mean
            entry = {"pass": it, "window": window, "turn": TURNS[window],
                     "distinct": count, "moved_to": route[window], "moved_from": old,
                     "confirm_mean": round(mean, 1), "was_tabu": was_tabu,
                     "improved_global": accepted, "route": pin_of(route)}
            log.append(entry)
            (out / "tabu.json").write_text(json.dumps(
                {"route": pin_of(best_route), "best_mean": round(best_mean, 1),
                 "log": log, "elapsed_sec": round(time.time() - t0)},
                ensure_ascii=False, indent=2), encoding="utf8")
            print(f"  窗口 {window}（T{TURNS[window]}）{old} → {route[window]}"
                  f"{'（禁忌期内靠渴望准则接受）' if was_tabu else ''} "
                  f"确认 {mean:.1f} "
                  f"{'↑全局最优' if accepted else '未超全局最优'}", flush=True)

    final = pin_of(best_route)
    print(f"\n禁忌搜索路线: {final}\n确认集均分 {best_mean:.1f}", flush=True)

    holdout = [args.holdout_seed + i for i in range(args.holdout_count)]

    def run(pin: str) -> list[dict]:
        return [play(binary, work, s, args.uma, args.cards, pin) for s in holdout]

    def summarize(rs: list[dict]) -> dict:
        ok = [r for r in rs if r["score"]]
        fives = [sum(r["five"]) for r in ok if r["five"]]
        lvs = [r["lv"] for r in ok if r["lv"]]
        return {
            "n": len(ok),
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

    print("验收：禁忌路线 …", flush=True)
    found = run(final)
    print("验收：坐标上升路线 …", flush=True)
    base = run(args.base_route)
    report = {"route_tabu": final, "route_coord": args.base_route,
              "holdout_seeds": f"{holdout[0]}-{holdout[-1]}",
              "tabu": summarize(found), "coord": summarize(base)}
    guide_runs: list[dict] = []
    if args.guide_route:
        print("验收：攻略路线 …", flush=True)
        gd = guide_runs = run(args.guide_route)
        report["route_guide"] = args.guide_route
        report["guide"] = summarize(gd)
    a = [r["score"] for r in found if r["score"]]
    for name, other in (("tabu_vs_coord", base), ("tabu_vs_guide", guide_runs)):
        b = [r["score"] for r in other if r["score"]]
        if not b:
            continue
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
