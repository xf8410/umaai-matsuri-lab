#!/usr/bin/env python3
"""真机口径对表：data/scenario_matsuri.json ⇄ lab/data/festa-master/master-data.json

`master-data.json` 由上游 umaai-project/FestaScenarioAnalyzer 的
`scripts/generate_festa_data.py` 从日服 `master.mdb` 只读导出（sha256 见同目录
`source.json`），覆盖 `single_mode_15_{act_info,advice,advice_effect,fusion_effect,
act_effect,schedule}` 六张表。本脚本把 lab 手写剧本数据里**已被解释成具名字段**的
每一格，逐一对回原始行，防止后续调参过程中把数据层改飘。

口径备注（对表时按此换算，不是数据不一致）：
  * `schedule.turn` 是 **0 基**回合（引擎 `check_point_index` 直接比 `base.turn`），
    真机 `turn_num` 是 1 基，所以 lab 值 = 真机值 − 1（23/47/67 ⇄ 24/48/68）。
  * 演目 Lv 效果里 `effect_type ∈ {201,106,107,109,206}` 全部带
    `condition_type=1 / condition_value=5`（编成支援卡类型 ≥5 种才发动），
    lab 由 `MatsuriGame::deck_has_five_types()` 承接；本脚本只核对「条件覆盖率」，
    发动逻辑在 Rust 侧。
  * 五围在 lab/引擎里是**未减半原始坐标**（`constants.rs`：1200 以上未减半）；
    分析器显示的 `Revise(x) = x>1200 ? 2x-1200 : x` 只是 HUD 折算，不进本表。

用法：
    python3 scripts/check_master_data.py            # 全量对表，不符则退出码 1
    python3 scripts/check_master_data.py --verbose   # 逐格打印
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB_DATA = ROOT / "data" / "scenario_matsuri.json"
MASTER = ROOT / "lab" / "data" / "festa-master" / "master-data.json"
SOURCE = ROOT / "lab" / "data" / "festa-master" / "source.json"

# 演目 Lv 效果：lab 具名字段 -> 真机 effect_type（数值都取 effect_value_4）
ACT_FIELDS = {
    "all_status_bonus": 105,   # 基礎能力の上昇量 +10
    "friendship": 202,         # 友情ボーナス N%
    "sp_training": 201,        # スキルPtのトレーニング効果アップ N%
    "sp_limit": 109,           # スキルPtの獲得上限 +N
    "status_limit": 206,       # 基礎能力の獲得上限 +N（Lv9/10 才有）
}
ACT_FLAGS = {
    "hint_event": 106,         # 全编成支援卡 Hint 事件发生
    "all_hint": 107,           # 所选训练的全 Hint 效果发动（Lv4 起）
}
# 这五类演目效果在真机上都挂「5 种卡」条件
ACT_CONDITIONAL = {201, 106, 107, 109, 206}

# 教诲：lab 具名字段 -> (effect_type, 取值方式)
#   "v4"      = effect_value_4
#   "neg_v4"  = -effect_value_4（体力消費ダウン存成负数）
#   "cond"    = condition_value（体力门槛型条件）
#   "flag"    = 只看该 effect_type 是否存在（lab 收成布尔）
ADVICE_FIELDS: dict[int, dict[str, tuple]] = {
    101: {"gauge": (16, "v4"), "keiko_main_bonus": (13, "v4")},          # 絆の教え
    201: {"deyilv": (4, "v4")},                                          # 得意率の教え
    301: {"hint_rate": (2, "v4")},                                       # ヒントの教え
    401: {"training_effect": (203, "v4"), "support_assist": (20, "flag")},  # 助力の教え
    501: {"keiko_vital_cost_drop": (3, "neg_v4")},                       # 体力の教え
    601: {"main_gain": (13, "v4"), "status_limit": (206, "v4")},         # 特化の教え
    701: {"rate": (8, "v4")},                                            # 研鑽の教え（追加の稽古）
    801: {"friendship": (202, "v4"), "split": (18, "v4")},               # 協力の教え
    901: {"fail_vital_threshold": (205, "cond"), "keiko_training_effect": (203, "v4")},
    1001: {"sub_gain": (21, "v4"), "keiko_training_effect": (203, "v4")},  # 両道の教え
}

# 転成附效：lab fusion[].effect 具名字段 -> (effect_type, 取值方式)
FUSION_FIELDS: dict[int, dict[str, tuple]] = {
    101: {"keiko_main_bonus": (13, "v4"), "camp_extra_person": (17, "v4")},
    201: {"deyilv": (4, "v4"), "keiko_friendship": (202, "v4")},
    301: {"hint_rate": (2, "v4"), "hint_level": (19, "v4")},
    401: {"training_effect": (203, "v4"), "support_assist": (20, "flag")},
    501: {"keiko_vital_cost_drop": (3, "neg_v4")},
}


class Report:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.ok = 0
        self.bad: list[str] = []
        self.notes: list[str] = []

    def cmp(self, where: str, got, want) -> None:
        if got == want:
            self.ok += 1
            if self.verbose:
                print(f"  ✓ {where} = {got}")
        else:
            self.bad.append(f"{where}: lab={got!r} 真机={want!r}")

    def note(self, text: str) -> None:
        self.notes.append(text)


def pick(rows: list[dict], **kw) -> list[dict]:
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def value_of(rows: list[dict], effect_type: int, how: str):
    """从真机行里取出与 lab 具名字段可比的值。"""
    hit = pick(rows, effect_type=effect_type)
    if how == "flag":
        return bool(hit)
    if not hit:
        return 0 if how in ("v4", "neg_v4") else None
    r = hit[0]
    if how == "v4":
        return r["effect_value_4"]
    if how == "neg_v4":
        return -r["effect_value_4"]
    if how == "cond":
        return r["condition_value"]
    raise ValueError(how)


def check_acts(rep: Report, lab: dict, tables: dict) -> None:
    real = tables["single_mode_15_act_info"]
    print("== 演目 act_info ==")
    for i, row in enumerate(lab["acts"]):
        r = real[i]
        rep.cmp(f"acts[{i}].act_type", row["act_type"], r["act_type"])
        rep.cmp(f"acts[{i}].command_id", row["command_id"], r["command_id"])
    rep.cmp("acts 条数", len(lab["acts"]), len(real))


def check_schedule(rep: Report, lab: dict, tables: dict) -> None:
    real = tables["single_mode_15_schedule"]
    print("== 発表会 schedule（lab 0 基 ⇄ 真机 1 基）==")
    rep.cmp("schedule 条数", len(lab["schedule"]), len(real))
    for i, row in enumerate(lab["schedule"]):
        r = real[i]
        rep.cmp(f"schedule[{i}].turn", row["turn"], r["turn_num"] - 1)
        rep.cmp(f"schedule[{i}].need_act_num", row["need_act_num"], r["need_act_num"])
        rep.cmp(f"schedule[{i}].need_act_level", row["need_act_level"], r["need_act_level"])


def check_act_effects(rep: Report, lab: dict, tables: dict) -> None:
    real = tables["single_mode_15_act_effect"]
    print("== 演目 Lv 效果 act_effect ==")
    rep.cmp("act_effects 级数", len(lab["act_effects"]), len({r["level"] for r in real}))
    for row in lab["act_effects"]:
        lv = row["level"]
        rows = pick(real, level=lv)
        for field, etype in ACT_FIELDS.items():
            rep.cmp(f"act_effects[Lv{lv}].{field}", row[field], value_of(rows, etype, "v4"))
        for field, etype in ACT_FLAGS.items():
            rep.cmp(f"act_effects[Lv{lv}].{field}", row[field], value_of(rows, etype, "flag"))
        # 「5 种卡」条件覆盖率：条件行必须全是 condition_type=1/value=5
        for r in rows:
            if r["effect_type"] in ACT_CONDITIONAL:
                rep.cmp(
                    f"act_effect id={r['id']} 的条件（5 种卡）",
                    (r["condition_type"], r["condition_value"]),
                    (1, 5),
                )
    cond = {(r["condition_type"], r["condition_value"]) for r in real}
    rep.note(f"act_effect 条件集合 = {sorted(cond)}（(1,5) = 编成支援卡 5 种以上）")


def check_advices(rep: Report, lab: dict, tables: dict) -> None:
    real_adv = tables["single_mode_15_advice"]
    real_eff = tables["single_mode_15_advice_effect"]
    print("== 教诲 advice / advice_effect ==")
    rep.cmp("advices 条数", len(lab["advices"]), len(real_adv))
    for i, row in enumerate(lab["advices"]):
        r = real_adv[i]
        rep.cmp(f"advices[{i}].id", row["id"], r["id"])
        rep.cmp(f"advices[{i}].fusion_type", row["fusion_type"], r["fusion_type"])
        rep.cmp(f"advices[{i}].add_schedule_type", row["add_schedule_type"], r["add_schedule_type"])
    for row in lab["advices"]:
        aid = row["id"]
        fields = ADVICE_FIELDS[aid]
        levels = row["effect"]["levels"]
        rep.cmp(f"advice {aid} 级数", len(levels), len({r["level"] for r in pick(real_eff, advice_id=aid)}))
        for i, lvrow in enumerate(levels, 1):
            rows = pick(real_eff, advice_id=aid, level=i)
            for field, (etype, how) in fields.items():
                if field not in lvrow:
                    continue
                want = value_of(rows, etype, how)
                if field == "support_assist":
                    # 真机用 effect_type 20 指向协助条件 id 47（助力の支え），lab 收成布尔
                    rep.cmp(f"advice {aid} Lv{i}.support_assist", bool(lvrow[field]), bool(want))
                else:
                    rep.cmp(f"advice {aid} Lv{i}.{field}", lvrow[field], want)


def check_fusion(rep: Report, lab: dict, tables: dict) -> None:
    real = tables["single_mode_15_fusion_effect"]
    print("== 転成附效 fusion_effect ==")
    by_advice = defaultdict(list)
    for r in real:
        by_advice[r["advice_id"]].append(r)
    rep.cmp("fusion 条数", len(lab["fusion"]), len(by_advice))
    for row in lab["fusion"]:
        aid = row["advice_id"]
        rows = by_advice.get(aid, [])
        for field, (etype, how) in FUSION_FIELDS.get(aid, {}).items():
            got = row["effect"].get(field)
            if got is None:
                continue
            want = value_of(rows, etype, how)
            if field == "support_assist":
                rep.cmp(f"fusion {aid}.support_assist", bool(got), bool(want))
            else:
                rep.cmp(f"fusion {aid}.{field}", got, want)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="逐格打印一致项")
    args = ap.parse_args()

    lab = json.loads(LAB_DATA.read_text(encoding="utf-8"))
    master = json.loads(MASTER.read_text(encoding="utf-8"))
    src = json.loads(SOURCE.read_text(encoding="utf-8"))
    tables = master["tables"]

    print(f"lab  : {LAB_DATA.relative_to(ROOT)}")
    print(f"真机 : {MASTER.relative_to(ROOT)}")
    print(f"来源 : {src['source']}  sha256={src['sha256'][:16]}…")
    print(f"表   : {src['tables']}\n")

    rep = Report(args.verbose)
    check_acts(rep, lab, tables)
    check_schedule(rep, lab, tables)
    check_act_effects(rep, lab, tables)
    check_advices(rep, lab, tables)
    check_fusion(rep, lab, tables)

    print()
    for n in rep.notes:
        print("· " + n)
    print(f"\n一致 {rep.ok} 格，不符 {len(rep.bad)} 格")
    for b in rep.bad:
        print("  ✗ " + b)
    return 1 if rep.bad else 0


if __name__ == "__main__":
    sys.exit(main())
