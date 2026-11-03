# 属性溢出摆烂 · 拉面杯「软上限储备」旋钮移植

> 来源：`xulai1001/umaai-rs`（拉面杯，master 分支）
> 目标：`xf8410/umaai-matsuri-lab`（骏大祭 / Festa，scenarioId 15）
> 范围：**只抄和「属性溢出」有关的旋钮**，其余（温泉挖掘 / 体力 / 比赛 / 友人外出 / 自选比赛摆烂门 / 诀窍槽溢出）一律不动。

---

## 1. 结论先行

拉面杯解决「属性溢出摆烂」靠的**不是**终局测量，而是评估器里一个**前瞻性的软上限边际函数**：

- 离上限远 → 训练边际**全额**计入；
- 进入「储备带」→ 边际按**二次曲线**折扣；
- 越过上限 → 边际**归零**。

关键在于**储备带宽度随剩余回合收缩**（`reserve = 40 × 剩余回合 × (1 − 剩余回合 /(2×总回合))`）：
前期储备带宽（~1500 点），逼 AI 把投资摊开、别把一维早早顶死；末期储备带→0，退化成硬上限，让 AI 把五维**同时贴满上限**（顶级人类面板的特征）。

matsuri 代码里**写了两套溢出机制，但当前配置下都没开**（实测 `data/matsuri_config.toml`）：

1. `effective_status_gain`：上限硬截断 + 超 `status_halve_threshold` 按 `status_halve_factor` 减半 —— 一个**固定阈值的悬崖**；当前 `status_halve_threshold = 0`（**减半关闭**）。
2. `dyn_status` / `status_catchup`：给「剩余空间占比高于均值」的轴加权 —— **与回合数无关**；当前无 `[trainer.dynamic]` 段（`status_catchup = 0`，**追赶关闭**）。

也就是说，**matsuri 现在实际生效的溢出处理只有硬截断 `.min(limit)` + 凸评分表**，没有任何「软折扣」或「时间感知」。这正是用户观察到「摆烂」的来源：一维顶到上限后边际突然归零（悬崖），AI 要么继续在死维上浪费回合、要么猛地甩到另一维。拉面那套「随时间收缩的软储备带」填的就是这个**真实空缺**（不是替换某个已开机制）。下面给出**逐旋钮源码 + 门控式移植补丁**，可直接 A/B。

---

## 2. 拉面在哪里解决溢出（定位）

| 关注点 | 文件 | 说明 |
|---|---|---|
| 软上限旋钮（本体） | `crates/umasim/src/neural/handwritten_evaluator.rs` | `status_soft_function` + `RESERVE_STATUS_FACTOR` + `FINAL_BONUS` + `calc_status_gain`，**全部硬编码常量，未进 toml** |
| 为什么终局测不出溢出 | `crates/umasim/src/search/ramen_terminal.rs:7-13` | `Uma::add_value` 每次 `.min(limit)` 当场截断，终局 `five_status` 不可能超上限，`max(0, status−limit)` 恒为 0 → 溢出只能在**决策层前瞻**，无法从终局反推 |
| 截断证据 | `crates/umasim/src/game/uma.rs:361-364` | `five_status[i] = (five_status[i] + status_pt[i]).min(five_status_limit[i])` |

> 注意：拉面的 `policy.rs` 里的「摆烂」是**自选比赛补不齐就不强制**（free-race gate），`rules.rs` 里的「溢出」是**诀窍槽（feeling gauge）溢出丢弃** —— 两者都**与属性溢出无关**，不在抄写范围。

---

## 3. 要抄的旋钮（仅溢出相关）

### 3.1 `status_soft_function` —— 软上限曲线（核心）

`handwritten_evaluator.rs:156-169`，**原样照抄**：

```rust
/// 属性上限软约束函数
/// x = 当前相对「有效上限」的超出量（x<0 表示仍有余量，x>=0 表示已到/超上限）
/// reserve = 软约束带宽（储备带）
fn status_soft_function(x: f64, reserve: f64) -> f64 {
    if reserve <= 0.0 {
        return x.min(0.0);            // 储备带为 0 → 退化成硬上限（只认负余量）
    }
    let reserve_inv_x2 = 1.0 / (2.0 * reserve);
    if x >= 0.0 {
        0.0                            // 已到/超上限 → 边际归零
    } else if x > -reserve {
        -x * x * reserve_inv_x2 * 0.5  // 储备带内 → 二次折扣 = -x²/(4·reserve)
    } else {
        x + 0.25 * reserve             // 远离上限 → 线性（≈全额）
    }
}
```

### 3.2 `RESERVE_STATUS_FACTOR` + 储备带公式（时间感知，关键创新）

`handwritten_evaluator.rs:52` 与 `:395-397`：

```rust
const RESERVE_STATUS_FACTOR: f64 = 40.0;   // 储备带因子

// calc_status_gain 内：
let remain_turn = (game.max_turn() - game.turn - 1).max(0) as f64;
let total_turn  = game.max_turn() as f64;
let reserve = RESERVE_STATUS_FACTOR * remain_turn * (1.0 - remain_turn / (total_turn * 2.0));
```

储备带随回合的取值（`MAX_TURN = 77`，matsuri 同口径）：

| 回合 | 剩余回合 | reserve（储备带宽） | 含义 |
|---|---|---|---|
| 0（开局） | 76 | ≈ 1538 | 带宽极宽：距上限 1500 点内就开始折扣 → 逼 AI 摊开五维 |
| 38（中段） | 38 | ≈ 1145 | 仍宽：继续抑制单维顶死 |
| 70（末段） | 6 | ≈ 231 | 收窄：允许把主力维推近上限 |
| 76（终局） | 0 | 0 | 退化成硬上限：五维同时贴满 |

### 3.3 `FINAL_BONUS` —— 终局事件预留

`handwritten_evaluator.rs:73`：

```rust
const FINAL_BONUS: i32 = 45 + 30 + 20 + 20; // = 115，URA3 + 最终事件 + URA2 + URA1
```

作用：把「有效上限」当成 `limit − FINAL_BONUS`，即**给终局事件的属性奖励预留空间**，避免决策层以为还能再吃 115 点而提前顶死。
⚠ **这是 URA/温泉剧本的终局事件加成，骏大祭（Festa）的终局事件不同，移植后必须按本剧本重标定**（见 §7）。

### 3.4 `calc_status_gain` 的接线（remain / s0 / s1）

`handwritten_evaluator.rs:394-417`，溢出相关骨架：

```rust
fn calc_status_gain(&self, game: &OnsenGame, train_value: &[i32; 6]) -> f64 {
    let remain_turn = (game.max_turn() - game.turn - 1).max(0) as f64;
    let total_turn  = game.max_turn() as f64;
    let reserve = RESERVE_STATUS_FACTOR * remain_turn * (1.0 - remain_turn / (total_turn * 2.0));

    let mut total = 0.0;
    for i in 0..5 {
        let limit  = game.uma.five_status_limit[i];
        let remain = limit - game.uma.five_status[i] - FINAL_BONUS;  // 距「有效上限」的余量
        let gain   = train_value[i];

        let s0 = status_soft_function(-remain as f64, reserve);          // 训练前的软约束值
        let s1 = status_soft_function((gain - remain) as f64, reserve);  // 训练后的软约束值
        let status_weight = if game.card_type_count[i] > 0 { STATUS_WEIGHTS[i] } else { ABSENT_WEIGHT };
        total += status_weight * (s1 - s0);   // 边际 = (s1 − s0)，远离上限时 ≈ gain
    }
    total += train_value[5] as f64 * self.skill_weight;  // PT
    total
}
```

> **`s1 − s0` 的量纲 = 属性点**：远离上限时 `s1−s0 = gain`（全额），储备带内 `< gain`（折扣），越上限 `= 0`。
> 这意味着它可以**直接当作「有效入账属性点」**喂回 matsuri 的凸评分表 —— 这是移植的接入点（见 §6）。

### 附：权重不是溢出旋钮（matsuri 已有对应，不必抄）

拉面 `STATUS_WEIGHTS=[7,8,8,8,6]`、`ABSENT_WEIGHT=2.0`、`skill_weight=0.5` 是**偏好权重**，matsuri 已有 `status_weights` / `dyn_status` / `pt_value` 对应，**保持 matsuri 自己的值**，不在「溢出旋钮」抄写范围。

---

## 4. 数学：为什么这样就不摆烂

设 `x = current + FINAL_BONUS − limit`（当前相对有效上限的超出量，负=有余量），`gain>0`：

- **远离上限**（`x ≤ −reserve`）：`s1−s0 = (x+gain) − x = gain` → **全额**，AI 正常堆这一维；
- **储备带内**（`−reserve < x < 0`）：`s1−s0 = gain` 被二次项削掉一块 → **边际递减**，AI 感到「再堆这一维越来越不值」，自然把回合让给别的维；
- **越上限**（`x ≥ 0`）：`s1−s0 = 0` → **归零**，AI 彻底不再选这一维（不浪费回合 = 不摆烂在死维上）。

而 `reserve` 随剩余回合**收缩**，把「该留多少余量」变成时间的函数：

- 前期 reserve 大 → 折扣带宽 → **抑制单维过早顶死**，逼出均衡投资；
- 后期 reserve→0 → 折扣带消失 → **放开手把五维同时推满上限**。

这正是「顶级人类终局五维同时贴近上限」面板的成因，也是 matsuri 现行**静态悬崖（halve）+ 时间无关追赶（catchup）**给不出的行为。

---

## 5. 与 matsuri 现状对比

| 维度 | matsuri 现行（实测当前配置） | 拉面软上限 | 差距 |
|---|---|---|---|
| 上限处理 | `effective_status_gain`：硬截断 + 超 `status_halve_threshold` 减半（`value.rs:615-632`）；**当前 threshold=0 → 减半关，只剩硬截断** | `status_soft_function`：二次软折扣 | matsuri 是**悬崖/硬墙**，拉面是**平滑带** |
| 时间感知 | 无（阈值固定） | `reserve` 随剩余回合收缩 | **matsuri 完全缺这一维** ← 要抄的核心 |
| 终局预留 | 无 | `FINAL_BONUS`（有效上限 = limit−FB） | matsuri 缺 |
| 五维均衡 | `status_catchup`/`dyn_status`：按剩余空间占比加权（`value.rs:182`），**与回合无关；当前 catchup=0 → 关** | 软储备带前期天然摊开 | 两者**部分重叠**，移植后需决定叠加还是二选一（§7） |
| 评分换算 | 凸表 `five_status_final_score`（非减半）× `status_weights`（当前全 1）× `dyn_status` | 线性权重 × `(s1−s0)` | matsuri 的凸表更细；移植时**保留凸表**，只把 `s1−s0` 当有效入账喂进去 |

matsuri 配置文档原话（`config.rs:96-101`）已点明其哲学：「官方评分表本身是凸的，贪心按边际打分已经能吃掉『越接近上限越值钱』，所以这里**不重复建模边际**」。
—— 拉面的软储备带恰恰是「**显式建模边际 + 时间感知**」，与 matsuri 哲学相反但互补；是否引入，用 A/B 数据说话。

---

## 6. 移植补丁（门控式，可直接 A/B）

设计原则：**保留 matsuri 的凸评分表与权重体系**，只把拉面的「软上限储备」作为 `effective_status_gain` 的一个**可切换分支**，用配置门控，默认关闭（不改变现有行为）。

### 6.1 `crates/matsuri-core/src/config.rs` —— 新增 3 个旋钮

在 `TrainerConfig` 里（紧邻 `status_halve_threshold` / `status_halve_factor`）加：

```rust
/// 属性溢出模式：false = 现行「上限截断 + 超阈值减半」；true = 拉面式「时间感知软上限储备」。
#[serde(default)]
pub status_soft_reserve: bool,

/// 软上限储备因子（拉面 RESERVE_STATUS_FACTOR = 40）。
/// reserve = factor × 剩余回合 × (1 − 剩余回合 /(2×总回合))；0 = 关闭软储备（退化硬上限）。
#[serde(default = "default_reserve_factor")]
pub reserve_status_factor: f64,

/// 终局事件预留属性（拉面 FINAL_BONUS = 115，URA 口径）。
/// 骏大祭终局事件不同，默认 0（不预留），重标定后再设。
#[serde(default)]
pub final_bonus: i32,
```

并加缺省函数 + 校验：

```rust
fn default_reserve_factor() -> f64 { 40.0 }

// 在 trainer 校验段（config.rs:400 附近）加：
checks.check(
    trainer.reserve_status_factor.is_finite() && (0.0..=200.0).contains(&trainer.reserve_status_factor),
    "trainer.reserve_status_factor 应在 0–200（0 = 关闭软储备）"
);
checks.check(
    (0..=600).contains(&trainer.final_bonus),
    "trainer.final_bonus 应在 0–600"
);
```

`Default for TrainerConfig` 里补：`status_soft_reserve: false, reserve_status_factor: 40.0, final_bonus: 0`。

### 6.2 `crates/matsuri-core/src/value.rs` —— 加软上限函数 + 分支

文件顶部（或 `status_gain` 附近）加**原样照抄**的软函数：

```rust
/// 拉面前属性上限软约束（handwritten_evaluator.rs::status_soft_function 原样移植）。
fn status_soft_function(x: f64, reserve: f64) -> f64 {
    if reserve <= 0.0 { return x.min(0.0); }
    let reserve_inv_x2 = 1.0 / (2.0 * reserve);
    if x >= 0.0 { 0.0 }
    else if x > -reserve { -x * x * reserve_inv_x2 * 0.5 }
    else { x + 0.25 * reserve }
}
```

把 `effective_status_gain`（`value.rs:615-632`）改成**门控双分支**：

```rust
fn effective_status_gain(&self, axis: usize, current: i32, gain: i32) -> i32 {
    let limit = self.game.uma().five_status_limit[axis];

    // —— 分支 A：拉面式时间感知软上限储备 ——
    if self.cfg.status_soft_reserve {
        let remain_turn = (self.game.max_turn() - self.game.turn() - 1).max(0) as f64;
        let total_turn  = self.game.max_turn().max(1) as f64;
        let reserve = self.cfg.reserve_status_factor * remain_turn
            * (1.0 - remain_turn / (total_turn * 2.0));
        let remain = limit - current - self.cfg.final_bonus;
        let s0 = status_soft_function(-remain as f64, reserve);
        let s1 = status_soft_function((gain - remain) as f64, reserve);
        // s1−s0 = 软折扣后的「有效入账属性点」，远离上限≈gain，越上限=0
        return ((s1 - s0).round() as i32).clamp(0, gain.max(0));
    }

    // —— 分支 B：现行「上限截断 + 超阈值减半」（保持不变）——
    let capped = ((current + gain).min(limit) - current).max(0);
    if capped == 0 || self.cfg.status_halve_threshold <= 0 {
        return capped;
    }
    let threshold = self.cfg.status_halve_threshold;
    let factor = self.cfg.status_halve_factor.clamp(0.0, 1.0);
    if current >= threshold {
        (f64::from(capped) * factor).round() as i32
    } else {
        let full = (threshold - current).min(capped);
        let over = capped - full;
        full + (f64::from(over) * factor).round() as i32
    }
}
```

> `status_gain`（`value.rs:591-609`）**不用改**：它已经把 `effective_status_gain` 的返回值当「有效入账」喂进凸表 `status_score(after) − status_score(current)`，再乘 `status_weights × dyn_status`。软分支只是换了「有效入账」的算法。

### 6.3 `data/matsuri_config.toml` —— A/B 开关

> 注意：trainer 配置在 **`data/matsuri_config.toml`**（不是 `ai_config.toml`，后者只有 `[sim]/[search]/[record]/[output]`）。当前实测值：`status_halve_threshold = 0`（减半关）、`status_weights = [1,1,1,1,1]`、无 `[trainer.dynamic]`（catchup 关）。

```toml
[trainer]
# 现行（对照组）—— 实际只有硬截断在生效：
status_soft_reserve  = false
status_halve_threshold = 0      # 现值（减半关）
status_halve_factor    = 0.5
status_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

# 拉面式（实验组）—— 打开软上限储备：
# status_soft_reserve   = true
# reserve_status_factor = 40.0
# final_bonus           = 0      # 待按骏大祭终局事件重标定
```

---

## 7. 骏大祭需重标定 / 注意的点

1. **`FINAL_BONUS` 必须重标定**：拉面的 115 = URA3+最终事件+URA2+URA1。骏大祭（Festa）的终局事件链不同，先设 `0`（不预留）跑通，再按本剧本终局事件的属性奖励实测填入。
2. **`reserve_status_factor=40` 是拉面/温泉口径**：温泉剧本总回合与属性上限和骏大祭不同（骏大祭 `five_status_limit_base=[3100,2400,2200,2200,2400]`，上限更高）。40 在骏大祭可能偏宽/偏窄，建议 A/B 扫 `{20, 40, 60}`。
3. **与 `status_catchup` 的关系**：两者都治「五维不均衡」，但机理不同（catchup=时间无关的占比加权；soft-reserve=时间感知的边际折扣）。移植后建议三臂对照：①仅 catchup（现行）②仅 soft-reserve ③catchup+soft-reserve，看是否冗余或互补。
4. **软函数在 `reserve=0` 时返回 `x.min(0)`**：终局回合 reserve→0，分支 A 退化成「只认负余量」，与硬截断一致，无副作用。
5. **量纲一致性**：`s1−s0` 返回的是**属性点**（非评分点），喂回凸表前不要乘权重；权重仍由 `status_gain` 里的 `status_weights × dyn_status` 负责。

---

## 8. 不要抄的（与属性溢出无关）

| 拉面里的东西 | 为什么不抄 |
|---|---|
| `evaluate_dig_value` / `DIG_*` / `RECOMMENDED_TOOL_TYPE` / 温泉选择 | 温泉挖掘，骏大祭无此机制 |
| `vital_evaluation` / `VITAL_FACTOR_*` / 超回复 / `should_skip_vital_penalty` | 体力与温泉超回复，matsuri 有自己的 `vital_curve` |
| `RACE_BASE_BONUS` / `FREE_RACE_BONUS` / 狄杜斯 1.8× | 比赛估值；且**生野狄杜斯是禁马**（gameIds 106301/106302） |
| `FRIEND_OUTING_*` / 友人外出超回复调整 | matsuri 已有 `friend_outing_deadline` / `friend_outing_urgency` |
| `policy.rs` 的「摆烂」free-race gate | 是「自选比赛补不齐就不强制」，**不是属性溢出** |
| `rules.rs` 的「溢出」诀窍槽（feeling gauge） | 是拉面剧本资源槽溢出丢弃，**不是属性溢出** |
| `STATUS_WEIGHTS` / `ABSENT_WEIGHT` / `skill_weight` | 偏好权重，matsuri 已有对应，保持 matsuri 自己的值 |

---

## 9. 验证建议（CRN A/B，≤100 局）

沿用 matsuri 的 CRN 配对（同 seed 跨臂，规则层随机一致，差异只来自配置）：

- **臂 0（对照）**：`status_soft_reserve=false`（现行 = 仅硬截断，halve/catchup 均关）。
- **臂 1**：`status_soft_reserve=true, reserve_status_factor=40, final_bonus=0`。
- **臂 2/3**：`reserve_status_factor ∈ {20, 60}` 扫带宽。
- 每臂 50 局（seeds 61444–61493），配对 t 检验。
- **看哪些指标**（沿用 standing 汇报口径）：育成评分、五维合计、五种属性分别、PT、稽古设施（演目）Lv；**额外盯**五维的 `headroom`（距上限）极差 —— 拉面软储备的目标就是把它压小（五维同时贴顶）。
- 朋友槽锁 SSR 吾妻、无根卡、禁生野狄杜斯、模拟器属性非减半（别和攻略游戏内面板比）—— 全部照旧。

> 若臂 1 评分不降且五维 headroom 极差收窄 → 软储备生效，可进一步调 `reserve_status_factor` 与 `final_bonus`；若评分下降 → 说明骏大祭的凸表 + catchup 已够，软储备冗余，保留门控默认关闭即可。
