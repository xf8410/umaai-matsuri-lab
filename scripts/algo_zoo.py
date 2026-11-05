#!/usr/bin/env python3
"""算法动物园：把「原始清单」里本仓还没跑过的优化器全部接到同一套评测口径上。

本仓已跑过（见 `lab/RESULTS.md` 轮次 7/8）：GA（两级保真）、贝叶斯优化（GP+EI）、CMA-ES、
模拟退火、禁忌搜索（离散路线）、UCB1 多臂（転成 120 组合）、坐标上升、PBT（ga/es/sa 算子）、
(1+1)-ES。交付包那轮另外跑过：SPSA 式零阶随机扰动、剂量-响应曲线、留一消融、死代码审计、
多保真粗筛→精评→holdout。

本脚本补的是**还没试过的**：
  pso        粒子群（全局速度 + 个体记忆，与 GA 的「只留排序信息」互补）
  de         差分进化 rand/1/bin（用种群内差分做步长，天然适应各维量纲差 3 个数量级）
  es11       (1+1)-ES + 1/5 成功法则（单点自适应，最省评估）
  spsa       同时扰动随机逼近（一次评估拿一个「随机方向导数」，34 维只需 2 局/步）
  nsga2      多目标（均分 ↑ / 尾部 worst-20% 均分 ↑）——差距在尾部，均值目标看不见它
  mapelites  质量-多样性（行为描述子 = 根演目Lv × 最强轴），保留一整个「流派档案」
  rsm        响应面 / 克里金式代理（二次多项式拟合已评点，取代理极值做候选）
  frace      F-Race 式序贯淘汰（Friedman 检验 + 逐步加种子），不是优化器而是**评测协议**，
             可套在任意优化器产生的候选批上（--race）

两个保真度（这是本轮最重要的口径发现）：
  off  搜索关（`MATSURI_NORMAL_SEARCH_N=0`）＝**手写档**，~0.15s/局。
       仓内 GA / BO / CMA-ES / PBT 全部跑在这一档 —— ga-r2 冠军层就是这么调出来的。
  on   搜索开（N=128 / topk=4 / radical=0，§9.4 定档），~13.4s/局＝**生产档**。
  先用 `--fidelity-probe` 量两档的秩相关：如果 off 档的排序不能预测 on 档，
  那么过去所有在 off 档上做的调参结论都要重估。

基因组 = ga.py 的 34 维 + overflow/endgame 7 维（两档都活）+ [search] 5 维（仅 on 档活）
       = 41 / 46 维。

用法：
  python3 scripts/algo_zoo.py --fidelity-probe 24          # 两档秩相关
  python3 scripts/algo_zoo.py --algo pso --gens 12 --pop 8 --fidelity off
  python3 scripts/algo_zoo.py --algo de  --gens 10 --pop 10 --fidelity on --race --eval-seeds 20
"""
import argparse
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from ga import BOUNDS, FUSION, genome_hash, paired_stats, parse_overlay  # noqa: E402

UMA = "100101"
CARDS = "303174,303114,303004,303224,303204,303234"
BIN = REPO / "target/release/matsuri"
import re  # noqa: E402
SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")

# ── 扩展基因：overflow / endgame（两档都活）+ [search]（仅 on 档活） ───────────────
EXTRA: dict[str, tuple[float, float, bool]] = {
    "overflow.margin": (0.0, 0.5, False),
    "overflow.tilt": (0.0, 4.0, False),
    "overflow.keep": (0.0, 2.0, False),
    "endgame.keiko": (0.0, 1000.0, False),
    "endgame.scarcity": (0.0, 4.0, False),
    "endgame.level": (0.0, 500.0, False),
    "endgame.normal_penalty": (0.0, 1000.0, False),
    # [search]：radical 已由 §9.4 定档 0（给 GA 留 0–50 的范围自己复证）
    "search.radical_factor_max": (0.0, 50.0, False),
    "search.search_group_size": (4.0, 512.0, True),
    "search.use_ucb": (0.0, 1.0, True),
    "search.search_cpuct": (0.0, 4.0, False),
    "search.topk": (2.0, 8.0, True),
}
OFF_KEYS = list(BOUNDS) + [k for k in EXTRA if not k.startswith("search.")]
ON_KEYS = list(BOUNDS) + list(EXTRA)
BOUNDS_ALL = {**BOUNDS, **EXTRA}

CACHE: dict[tuple, dict] = {}


# ── 覆盖层生成 ──────────────────────────────────────────────────────────────────
def genome_to_toml(g: dict) -> str:
    """41/46 维基因组 → matsuri_config.toml 覆盖层（[trainer] 整段替换，必须给全键）。"""
    sw = ", ".join(f"{g[f'status_weights.{i}']:.4f}" for i in range(5))
    out = [FUSION, "[trainer]",
           f"pt_favor_rate = {g['pt_favor_rate']:.4f}",
           f"rest_vital = {int(g['rest_vital'])}",
           f"checkpoint_margin = {int(g['checkpoint_margin'])}",
           f"act_level_target = {int(g['act_level_target'])}",
           "self_race = false", "friend_outing_early = true",
           f"status_halve_threshold = {int(g['status_halve_threshold'])}",
           "status_halve_factor = 0.5", "friend_outing_deadline = 0",
           f"friend_outing_urgency = {g['friend_outing_urgency']:.4f}",
           "level_step_bonus = 0.0", f"status_weights = [{sw}]", "",
           "[trainer.weights]"]
    for k in ("vital", "motivation", "friendship", "hint", "gauge", "act_level",
              "checkpoint", "fat", "friend_outing"):
        out.append(f"{k} = {g[f'weights.{k}']:.4f}")
    out += ["", "[trainer.advice]"]
    for k in ("card", "npc", "train", "keiko", "level", "milestone", "status_limit",
              "assist", "route"):
        out.append(f"{k} = {g[f'advice.{k}']:.4f}")
    out += ["", "[trainer.dynamic]"]
    for k in ("status_catchup", "act_catchup", "vital_decay", "vital_idle", "keiko_tempo"):
        out.append(f"{k} = {g[f'dynamic.{k}']:.4f}")
    out += ["", "[trainer.overflow]",
            f"margin = {g['overflow.margin']:.4f}", f"tilt = {g['overflow.tilt']:.4f}",
            f"keep = {g['overflow.keep']:.4f}", "", "[trainer.endgame]",
            f"keiko = {g['endgame.keiko']:.4f}", f"scarcity = {g['endgame.scarcity']:.4f}",
            f"level = {g['endgame.level']:.4f}",
            f"normal_penalty = {g['endgame.normal_penalty']:.4f}", ""]
    return "\n".join(out)


def play(g: dict, seed: int, fidelity: str) -> dict:
    on = fidelity == "on"
    search = []
    if on:
        search = ["search_n = 128",
                  f"radical_factor_max = {g['search.radical_factor_max']:.4f}",
                  f"search_group_size = {int(g['search.search_group_size'])}",
                  f"use_ucb = {'true' if g['search.use_ucb'] > 0.5 else 'false'}",
                  f"search_cpuct = {g['search.search_cpuct']:.4f}"]
    (REPO / "matsuri_config.toml").write_text(genome_to_toml(g), encoding="utf8")
    (REPO / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {UMA}\ncards = [{CARDS.replace(',', ', ')}]\n"
        + (f"[search]\n" + "\n".join(search) + "\n" if search else ""), encoding="utf8")
    env = {**os.environ,
           "MATSURI_NORMAL_SEARCH_N": "128" if on else "0",
           "MATSURI_NORMAL_SEARCH_TOPK": str(int(g["search.topk"])) if on else "4"}
    t0 = time.time()
    try:
        p = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                           text=True, errors="replace", timeout=7200, env=env)
    except subprocess.TimeoutExpired:
        return {"score": 0, "secs": round(time.time() - t0, 1), "err": "timeout"}
    out, err = p.stdout, p.stderr
    m = SCORE_RE.search(out)
    rec = {"secs": round(time.time() - t0, 1)}
    if not m:
        return {**rec, "score": 0, "err": (out[-200:] + err[-200:]).strip()}
    five, lv = FIVE_RE.search(out), LV_RE.search(err)
    rec["score"] = int(m.group(1))
    rec["pt"] = int(PT_RE.search(out).group(1)) if PT_RE.search(out) else 0
    rec["five"] = [int(x) for x in five.groups()] if five else []
    rec["lv"] = [int(x) for x in lv.group(1).split(",")] if lv else []
    return rec


def evaluate(g: dict, seeds: list[int], fidelity: str, use_cache: bool = True) -> dict:
    """CRN 配对评测：逐种子跑，缓存 (基因组, 种子, 保真度) 级结果，racing 复用不重跑。"""
    gh = genome_hash({k: round(v, 6) for k, v in g.items() if k in BOUNDS_ALL})
    scores, pts, lvs, fives, games = [], [], [], [], []
    for s in seeds:
        key = (gh, s, fidelity)
        rec = CACHE.get(key)
        if rec is None:
            rec = play(g, s, fidelity)
            if use_cache:
                CACHE[key] = rec
        if rec["score"] == 0:
            continue
        games.append({"seed": s, **rec})
        scores.append(rec["score"])
        pts.append(rec.get("pt", 0))
        if rec.get("lv"):
            lvs.append(rec["lv"])
        if rec.get("five"):
            fives.append(rec["five"])
    if not scores:
        return {"fitness": 0.0, "tail": 0.0, "n": 0, "games": []}
    n = len(scores)
    k = max(1, n // 5)
    return {"fitness": sum(scores) / n,
            "tail": sum(sorted(scores)[:k]) / k,      # worst-20% 均分（CVaR 下侧）
            "n": n, "min": min(scores), "max": max(scores),
            "pt": sum(pts) / n,
            "lv": [sum(x[i] for x in lvs) / len(lvs) for i in range(5)] if lvs else [],
            "root_lv": (sum(x[3] for x in lvs) / len(lvs)) if lvs else 0.0,
            "top_axis": (max(range(5), key=lambda i: sum(x[i] for x in fives))
                         if fives else 0),
            "games": games}


# ── 基因组工具 ──────────────────────────────────────────────────────────────────
def keys(fidelity: str) -> list[str]:
    return ON_KEYS if fidelity == "on" else OFF_KEYS


def clip(g: dict, fidelity: str) -> dict:
    out = dict(g)
    for k in keys(fidelity):
        lo, hi, is_int = BOUNDS_ALL[k]
        out[k] = min(hi, max(lo, out[k]))
        if is_int:
            out[k] = float(round(out[k]))
    return out


def parse_overlay_ext(text: str) -> dict:
    """ga.parse_overlay + `[trainer.overflow]` / `[trainer.endgame]` 两段（ga.py 不认）。"""
    g = dict(parse_overlay(text))
    section = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            section = s.strip("[]")
            continue
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, value = s.partition("=")
        full = {"trainer.overflow": f"overflow.{key.strip()}",
                "trainer.endgame": f"endgame.{key.strip()}"}.get(section)
        if full in EXTRA:
            try:
                g[full] = float(value.strip())
            except ValueError:
                pass
    return g


def seed_genome(fidelity: str, base: Path) -> dict:
    """起点 = ga-r2 冠军层（能读到的键）+ 扩展键的默认层值。"""
    g = {}
    text = base.read_text(encoding="utf8")
    flat = parse_overlay_ext(text)
    for k in keys(fidelity):
        if k in flat:
            g[k] = flat[k]
        elif k.startswith("search."):
            g[k] = SEARCH_DEFAULTS[k]
        else:
            g[k] = 0.0    # overflow / endgame 缺省 = 关
    return clip(g, fidelity)


def rand_genome(fidelity: str, rng: random.Random) -> dict:
    return clip({k: rng.uniform(*BOUNDS_ALL[k][:2]) for k in keys(fidelity)}, fidelity)


SEARCH_DEFAULTS = {"search.radical_factor_max": 0.0, "search.search_group_size": 256.0,
                   "search.use_ucb": 1.0, "search.search_cpuct": 1.0, "search.topk": 4.0}


def with_search_keys(g: dict) -> dict:
    """补全 `[search]` 5 键：保真度探针要拿同一个基因组同时跑 off / on 两档，
    off 档的基因组里没有这几键，直接送进 on 档会 KeyError。"""
    out = dict(g)
    for k, v in SEARCH_DEFAULTS.items():
        out.setdefault(k, v)
    return out


def vec(g: dict, fidelity: str) -> list[float]:
    return [g[k] for k in keys(fidelity)]


def unvec(v: list[float], fidelity: str, base: dict) -> dict:
    g = dict(base)
    g.update({k: x for k, x in zip(keys(fidelity), v)})
    return clip(g, fidelity)


def unit(v: list[float], fidelity: str) -> list[float]:
    """归一到单位立方体（各维量纲差 3 个数量级，PSO/DE/RSM 必须在单位空间里走）。"""
    return [(x - BOUNDS_ALL[k][0]) / max(1e-9, BOUNDS_ALL[k][1] - BOUNDS_ALL[k][0])
            for k, x in zip(keys(fidelity), v)]


def ununit(u: list[float], fidelity: str, base: dict) -> dict:
    return unvec([BOUNDS_ALL[k][0] + x * (BOUNDS_ALL[k][1] - BOUNDS_ALL[k][0])
                  for k, x in zip(keys(fidelity), u)], fidelity, base)


# ── F-Race：序贯淘汰（Friedman + 逐步加种子） ───────────────────────────────────
def friedman_drop(rows: dict[int, list[float]]) -> list[int]:
    """rows: 候选 id → 逐种子得分（长度相同）。返回「显著劣于最优」的候选 id 列表。

    简化 F-Race：按每个种子内的排名做 Friedman 统计量，若显著则用最优者做 Bonferroni
    两两比较，把显著劣于最优的候选剔掉（保守：只剔显著劣的，不剔并列的）。
    """
    ids = sorted(rows)
    n_seeds = min(len(rows[i]) for i in ids)
    k = len(ids)
    if k < 3 or n_seeds < 3:
        return []
    ranks = {i: [0.0] * n_seeds for i in ids}
    for s in range(n_seeds):
        order = sorted(ids, key=lambda i: -rows[i][s])
        r = 0
        while r < k:
            tie = [order[r]]
            while r + 1 < k and rows[order[r + 1]][s] == rows[order[r]][s]:
                r += 1
                tie.append(order[r])
            avg = (r - len(tie) + 1 + r + 1) / 2.0
            for i in tie:
                ranks[i][s] = avg
            r += 1
    rbar = {i: sum(ranks[i]) / n_seeds for i in ids}
    chi = 12 * n_seeds / (k * (k + 1)) * sum((rbar[i] - (k + 1) / 2) ** 2 for i in ids)
    if chi < k - 1:                     # 未过 Friedman 临界（α≈0.05 的粗近似）
        return []
    best = min(ids, key=lambda i: rbar[i])
    se = math.sqrt(k * (k + 1) / (6.0 * n_seeds))
    crit = 1.96 + 2.58 / math.sqrt(k)    # Bonferroni 粗校正
    return [i for i in ids if i != best and (rbar[i] - rbar[best]) > crit * se]


def race(cands: list[dict], seed_pool: list[int], fidelity: str, step: int = 4,
         max_seeds: int = 20, log=print) -> list[tuple[dict, dict]]:
    """对一批候选做 F-Race：每轮加 step 个种子，剔掉显著劣的，直到只剩 ≤2 个或用满预算。"""
    alive = list(range(len(cands)))
    used = 0
    rows: dict[int, list[float]] = {i: [] for i in alive}
    while used < min(max_seeds, len(seed_pool)):
        take = seed_pool[used:used + step]
        used += len(take)
        for i in alive:
            for s in take:
                r = evaluate(cands[i], [s], fidelity)
                rows[i].append(r["fitness"] if r["n"] else 0.0)
        drop = friedman_drop({i: rows[i] for i in alive})
        if drop:
            alive = [i for i in alive if i not in drop]
            log(f"    race@{used}局 剔除 {len(drop)} 个，剩 {len(alive)}")
        if len(alive) <= 2:
            break
    return [(cands[i], evaluate(cands[i], seed_pool[:used], fidelity)) for i in alive]


# ── 优化器 ──────────────────────────────────────────────────────────────────────
def opt_pso(base, fidelity, pop, gens, rng, screen, log):
    ks = keys(fidelity)
    P = [[rng.random() for _ in ks] for _ in range(pop)]
    P[0] = unit(vec(base, fidelity), fidelity)   # 0 号粒子 = 冠军起点
    V = [[rng.uniform(-0.1, 0.1) for _ in ks] for _ in range(pop)]
    Pb = [list(p) for p in P]
    Pb_f = [None] * pop
    G = list(P[0])
    G_f = None
    w, c1, c2 = 0.729, 1.494, 1.494
    for gen in range(gens):
        for i in range(pop):
            g = ununit(P[i], fidelity, base)
            r = evaluate(g, screen, fidelity)
            if Pb_f[i] is None or r["fitness"] > Pb_f[i]:
                Pb_f[i], Pb[i] = r["fitness"], list(P[i])
                if G_f is None or r["fitness"] > G_f:
                    G_f, G = r["fitness"], list(P[i])
            for d in range(len(ks)):
                V[i][d] = (w * V[i][d] + c1 * rng.random() * (Pb[i][d] - P[i][d])
                           + c2 * rng.random() * (G[d] - P[i][d]))
                V[i][d] = max(-0.5, min(0.5, V[i][d]))
                P[i][d] = max(0.0, min(1.0, P[i][d] + V[i][d]))
        log(f"  [pso] gen{gen} 全局 {G_f:.1f}")
    return ununit(G, fidelity, base), G_f


def opt_de(base, fidelity, pop, gens, rng, screen, log):
    ks = keys(fidelity)
    P = [unit(vec(base, fidelity), fidelity) if i == 0
         else [rng.random() for _ in ks] for i in range(pop)]
    F = [(evaluate(ununit(p, fidelity, base), screen, fidelity)["fitness"]) for p in P]
    for gen in range(gens):
        for i in range(pop):
            a, b, c = rng.sample([j for j in range(pop) if j != i], 3)
            d0 = rng.randrange(len(ks))
            trial = list(P[i])
            for d in range(len(ks)):
                if rng.random() < 0.9 or d == d0:
                    trial[d] = max(0.0, min(1.0, P[a][d] + 0.6 * (P[b][d] - P[c][d])))
            tf = evaluate(ununit(trial, fidelity, base), screen, fidelity)["fitness"]
            if tf >= F[i]:
                P[i], F[i] = trial, tf
        best = max(range(pop), key=lambda i: F[i])
        log(f"  [de] gen{gen} 最优 {F[best]:.1f}")
    best = max(range(pop), key=lambda i: F[i])
    return ununit(P[best], fidelity, base), F[best]


def opt_es11(base, fidelity, pop, gens, rng, screen, log):
    """(1+1)-ES：1/5 成功法则自适应步长（pop 参数忽略，单点搜索）。"""
    cur = dict(base)
    cur_f = evaluate(cur, screen, fidelity)["fitness"]
    sigma = 0.08
    for gen in range(gens):
        ok = 0
        trials = max(4, pop)
        for _ in range(trials):
            cand = dict(cur)
            for k in keys(fidelity):
                if rng.random() < 1.0 / len(keys(fidelity)) * 3:
                    lo, hi, _i = BOUNDS_ALL[k]
                    cand[k] = cur[k] + rng.gauss(0.0, sigma * (hi - lo))
            cand = clip(cand, fidelity)
            f = evaluate(cand, screen, fidelity)["fitness"]
            if f > cur_f:
                cur, cur_f, ok = cand, f, ok + 1
        sigma *= 1.15 if ok >= trials / 5 else 0.85
        sigma = max(0.01, min(0.4, sigma))
        log(f"  [es11] gen{gen} {cur_f:.1f} σ={sigma:.3f} 命中 {ok}/{trials}")
    return cur, cur_f


def opt_spsa(base, fidelity, pop, gens, rng, screen, log):
    """SPSA：一步只用 2 次评测拿一个随机方向导数（对称差分），a/c 按标准衰减。"""
    th = unit(vec(base, fidelity), fidelity)
    a, c, A, alpha, gamma = 0.6, 0.02, 4, 0.602, 0.101
    best_f = evaluate(base, screen, fidelity)["fitness"]
    best = dict(base)
    for gen in range(gens):
        ak = a / (gen + 1 + A) ** alpha
        ck = c / (gen + 1) ** gamma
        d = [1 if rng.random() < 0.5 else -1 for _ in th]
        gp = ununit([min(1, max(0, x + ck * dd)) for x, dd in zip(th, d)], fidelity, base)
        gm = ununit([min(1, max(0, x - ck * dd)) for x, dd in zip(th, d)], fidelity, base)
        fp = evaluate(gp, screen, fidelity)["fitness"]
        fm = evaluate(gm, screen, fidelity)["fitness"]
        grad = (fp - fm) / (2 * ck)
        th = [min(1.0, max(0.0, x - ak * grad * dd)) for x, dd in zip(th, d)]
        g = ununit(th, fidelity, base)
        f = evaluate(g, screen, fidelity)["fitness"]
        if f > best_f:
            best_f, best = f, g
        log(f"  [spsa] gen{gen} 新点 {f:.1f}（{fp:.0f}/{fm:.0f}）最优 {best_f:.1f}")
    return best, best_f


def opt_nsga2(base, fidelity, pop, gens, rng, screen, log):
    """NSGA-II：目标 = (均分 ↑, worst-20% 尾部均分 ↑)；快速非支配排序 + 拥挤度。"""
    P = [dict(base)] + [rand_genome(fidelity, rng) for _ in range(pop - 1)]
    for gen in range(gens):
        Q = []
        for p in P:
            a, b = rng.sample(P, 2)
            child = {k: (a[k] if rng.random() < 0.5 else b[k]) for k in keys(fidelity)}
            for k in keys(fidelity):
                if rng.random() < 0.15:
                    lo, hi, _ = BOUNDS_ALL[k]
                    child[k] += rng.gauss(0, 0.12 * (hi - lo))
            Q.append(clip(child, fidelity))
        R = P + Q
        objs = []
        for g in R:
            r = evaluate(g, screen, fidelity)
            objs.append((r["fitness"], r["tail"]))
        fronts = fast_fronts(objs)
        newP: list[dict] = []
        for fr in fronts:
            if len(newP) + len(fr) <= pop:
                newP += [R[i] for i in fr]
            else:
                crowd = crowding([objs[i] for i in fr])
                order = sorted(range(len(fr)), key=lambda j: -crowd[j])
                newP += [R[fr[j]] for j in order[:pop - len(newP)]]
                break
        P = newP
        f0 = max(objs[i][0] for i in range(len(R)))
        t0 = max(objs[i][1] for i in range(len(R)))
        log(f"  [nsga2] gen{gen} 前沿均分 {f0:.1f} / 尾部 {t0:.1f}")
    best = max(P, key=lambda g: evaluate(g, screen, fidelity)["fitness"])
    r = evaluate(best, screen, fidelity)
    return best, r["fitness"]


def fast_fronts(objs: list[tuple[float, float]]) -> list[list[int]]:
    n = len(objs)
    dom = [[] for _ in range(n)]
    cnt = [0] * n
    for i, j in itertools.combinations(range(n), 2):
        a, b = objs[i], objs[j]
        if a[0] >= b[0] and a[1] >= b[1] and (a[0] > b[0] or a[1] > b[1]):
            dom[i].append(j)
            cnt[j] += 1
        elif b[0] >= a[0] and b[1] >= a[1] and (b[0] > a[0] or b[1] > a[1]):
            dom[j].append(i)
            cnt[i] += 1
    fronts, cur = [], [i for i in range(n) if cnt[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in dom[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def crowding(pts: list[tuple[float, float]]) -> list[float]:
    n = len(pts)
    d = [0.0] * n
    for m in (0, 1):
        order = sorted(range(n), key=lambda i: pts[i][m])
        lo, hi = pts[order[0]][m], pts[order[-1]][m]
        span = hi - lo or 1.0
        d[order[0]] = d[order[-1]] = float("inf")
        for j in range(1, n - 1):
            d[order[j]] += (pts[order[j + 1]][m] - pts[order[j - 1]][m]) / span
    return d


def opt_mapelites(base, fidelity, pop, gens, rng, screen, log):
    """MAP-Elites：行为描述子 = (根演目Lv 分箱 0–3, 最强轴 0–4) → 20 格档案。

    与单目标优化器不同，它保留**一整个流派档案**：万一「根演目 Lv 7.2 结构缺口」
    只有某个非主流流派能突破，均值目标的优化器会把它淘汰掉，这里不会。
    """
    def bd(r):
        rl = r.get("root_lv", 0.0)
        b0 = 0 if rl < 7 else 1 if rl < 8.5 else 2 if rl < 9.5 else 3
        return (b0, r.get("top_axis", 0))
    archive: dict[tuple, tuple[dict, float]] = {}
    P = [dict(base)] + [rand_genome(fidelity, rng) for _ in range(pop - 1)]
    for gen in range(gens):
        for g in P:
            r = evaluate(g, screen, fidelity)
            cell = bd(r)
            if cell not in archive or r["fitness"] > archive[cell][1]:
                archive[cell] = (g, r["fitness"])
        P = []
        while len(P) < pop:
            parent = rng.choice(list(archive.values()))[0]
            child = dict(parent)
            for k in keys(fidelity):
                if rng.random() < 3.0 / len(keys(fidelity)):
                    lo, hi, _ = BOUNDS_ALL[k]
                    child[k] += rng.gauss(0, 0.15 * (hi - lo))
            P.append(clip(child, fidelity))
        log(f"  [mapelites] gen{gen} 档案 {len(archive)} 格，最优 "
            f"{max(v[1] for v in archive.values()):.1f}")
    best = max(archive.values(), key=lambda v: v[1])
    return best[0], best[1]


def opt_rsm(base, fidelity, pop, gens, rng, screen, log):
    """响应面 / 克里金式代理：单位空间里拟合「线性 + 平方」二次面，取代理极值做候选。

    样本 < 参数数时用岭正则最小二乘（不引 numpy 以外的依赖；numpy 缺失则退化到随机采点）。
    """
    pts, ys = [], []
    X = [unit(vec(base, fidelity), fidelity)]
    for _ in range(max(pop, len(keys(fidelity)) // 2)):
        X.append([rng.random() for _ in keys(fidelity)])
    for gen in range(gens):
        for u in X:
            g = ununit(u, fidelity, base)
            y = evaluate(g, screen, fidelity)["fitness"]
            pts.append(u)
            ys.append(y)
        X = []
        beta = fit_quad(pts, ys, len(keys(fidelity)))
        for _ in range(pop):
            if beta is None:
                X.append([rng.random() for _ in keys(fidelity)])
                continue
            u = surrogate_argmax(beta, len(keys(fidelity)), rng)
            X.append(u)
        log(f"  [rsm] gen{gen} 样本 {len(pts)}，实测最优 {max(ys):.1f}")
    best_u = pts[max(range(len(ys)), key=lambda i: ys[i])]
    return ununit(best_u, fidelity, base), max(ys)


def fit_quad(pts, ys, d):
    try:
        import numpy as np
    except ImportError:
        return None
    def feat(u):
        return [1.0] + list(u) + [u[i] ** 2 for i in range(d)]
    A = np.array([feat(u) for u in pts])
    y = np.array(ys)
    lam = 1e-3 * len(pts)
    try:
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    except np.linalg.LinAlgError:
        return None
    return beta


def surrogate_argmax(beta, d, rng, tries: int = 4000):
    import numpy as np
    best, bu = -1e18, None
    for _ in range(tries):
        u = np.clip(rng.gauss(0.5, 0.25, d), 0, 1)
        val = beta[0] + beta[1:1 + d] @ u + beta[1 + d:] @ (u ** 2)
        if val > best:
            best, bu = val, u
    return [float(x) for x in bu]


OPTS = {"pso": opt_pso, "de": opt_de, "es11": opt_es11, "spsa": opt_spsa,
        "nsga2": opt_nsga2, "mapelites": opt_mapelites, "rsm": opt_rsm}


# ── 保真度探针：off 档能不能预测 on 档 ─────────────────────────────────────────
def fidelity_probe(n: int, base: dict, rng: random.Random, eval_seeds: list[int],
                   out: Path) -> int:
    """同一批基因组在 off / on 两档各评一遍，算 Spearman 秩相关 + 配对差。

    这是本轮方法论上最要紧的一问：仓内 GA / BO / CMA-ES / PBT **全部**跑在 off 档
    （ga.py 的 play() 不设 MATSURI_NORMAL_SEARCH_N → 搜索关），而生产是 on 档。
    如果两档秩相关低，那 ga-r2 冠军层对生产档而言就是「在别的题目上考出来的高分」。
    """
    cands = [with_search_keys(base)] + [with_search_keys(rand_genome("off", rng))
                                        for _ in range(n - 1)]
    rows = []
    for i, g in enumerate(cands):
        off = evaluate(g, eval_seeds, "off")
        on = evaluate(g, eval_seeds, "on")
        rows.append({"i": i, "off": off["fitness"], "on": on["fitness"],
                     "off_tail": off["tail"], "on_tail": on["tail"],
                     "genome": {k: round(v, 4) for k, v in g.items() if k in OFF_KEYS}})
        print(f"  [{i + 1}/{n}] off {off['fitness']:9.1f}  on {on['fitness']:9.1f}  "
              f"({sum(r['secs'] for r in on['games']):.0f}s)", flush=True)

    def spearman(xs, ys):
        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            for pos, i in enumerate(o):
                r[i] = pos
            return r
        rx, ry = rank(xs), rank(ys)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else 0.0

    res = {"n_cands": n, "eval_seeds": eval_seeds,
           "spearman_off_on": round(spearman([r["off"] for r in rows],
                                             [r["on"] for r in rows]), 4),
           "spearman_offtail_ontail": round(spearman([r["off_tail"] for r in rows],
                                                     [r["on_tail"] for r in rows]), 4),
           "pearson_off_on": round(sum((r["off"] - sum(x["off"] for x in rows) / n)
                                       * (r["on"] - sum(x["on"] for x in rows) / n)
                                       for r in rows) / n
                                   / (math.sqrt(sum((r["off"] - sum(x["off"] for x in rows) / n) ** 2
                                                    for r in rows) / n)
                                      * math.sqrt(sum((r["on"] - sum(x["on"] for x in rows) / n) ** 2
                                                      for r in rows) / n) + 1e-9), 4),
           "rows": rows}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                     ensure_ascii=False, indent=1))
    return 0


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="pso", choices=sorted(OPTS))
    ap.add_argument("--fidelity", default="off", choices=("off", "on"))
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--screen-seeds", default="73601-73612", help="搜索期种子块")
    ap.add_argument("--holdout-seeds", default="73701-73720", help="留出裁决种子块")
    ap.add_argument("--race", action="store_true", help="候选批走 F-Race 序贯淘汰")
    ap.add_argument("--base-config", default=str(REPO / "lab/config/ga-r2.toml"))
    ap.add_argument("--binary", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fidelity-probe", type=int, default=0, help="跑保真度探针（候选数）")
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args()

    global BIN
    if args.binary:
        BIN = Path(args.binary)
    rng = random.Random(args.seed)
    base = seed_genome("on" if args.fidelity == "on" else "off", Path(args.base_config))
    lo, hi = (int(x) for x in args.screen_seeds.split("-"))
    screen = list(range(lo, hi + 1))
    hlo, hhi = (int(x) for x in args.holdout_seeds.split("-"))
    holdout = list(range(hlo, hhi + 1))

    if args.fidelity_probe:
        return fidelity_probe(args.fidelity_probe, base, rng, screen[:8],
                              Path(args.out or "/tmp/fidelity_probe.json"))

    out = Path(args.out or f"/tmp/zoo_{args.algo}_{args.fidelity}.json")
    t0 = time.time()
    b = evaluate(base, screen, args.fidelity)
    print(f"起点 ga-r2：{b['fitness']:.1f}（尾部 {b['tail']:.1f}，{b['n']} 局）", flush=True)
    best, best_f = OPTS[args.algo](base, args.fidelity, args.pop, args.gens, rng,
                                   screen, lambda m: print(m, flush=True))
    if args.race:
        cands = [best, base]
        surv = race(cands, screen, args.fidelity, log=lambda m: print(m, flush=True))
        best, _ = max(surv, key=lambda x: x[1]["fitness"])
    hold = evaluate(best, holdout, args.fidelity)
    hold_base = evaluate(base, holdout, args.fidelity)
    d, t, w = paired_stats([g["score"] for g in hold["games"]],
                           [g["score"] for g in hold_base["games"]])
    res = {"algo": args.algo, "fidelity": args.fidelity, "pop": args.pop,
           "gens": args.gens, "screen_seeds": args.screen_seeds,
           "holdout_seeds": args.holdout_seeds, "race": args.race,
           "screen_best": best_f, "screen_base": b["fitness"],
           "holdout_best": hold["fitness"], "holdout_base": hold_base["fitness"],
           "holdout_delta": d, "holdout_t": t, "holdout_wins": w,
           "holdout_tail_best": hold["tail"], "holdout_tail_base": hold_base["tail"],
           "games": len(CACHE), "secs": round(time.time() - t0, 1),
           "best_genome": {k: round(v, 4) for k, v in best.items()},
           "best_toml": genome_to_toml(best)}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({k: v for k, v in res.items() if k != "best_toml"},
                     ensure_ascii=False, indent=1))
    for f in ("matsuri_config.toml", "ai_config.toml"):
        (REPO / f).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
