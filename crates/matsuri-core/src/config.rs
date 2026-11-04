//! 剧本配置段（`matsuri_config.toml`）：schema、载入与覆盖层合并
//!
//! 复用引擎「默认层 + 覆盖层」的泛型解析（`OverlayConfig` / `parse_with_overlay`）：
//! 默认层字段齐全、未知键严格报错；覆盖层同构、字段全 `Option`，段缺失整段沿用。
//! 根目录由宿主注入，本模块不写死路径。
//!
//! 当前只放**融合（転成）预设**：默认方案固定、在线可覆盖一次（融合不进搜索树），
//! 训练员 / 搜索参数留 S7 / M2，不预置。

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use umaai::config::{OverlayConfig, parse_with_overlay};

/// 剧本配置文件名（放在宿主注入的剧本配置根下）
pub const MATSURI_CONFIG_FILE: &str = "matsuri_config.toml";

/// 基础教诲 id（融合的 `base` 只能是这 5 个，且每个只能用一次）
const BASE_ADVICE_IDS: [i32; 5] = [101, 201, 301, 401, 501];

/// 第 2 年（turn 23）可用的融合对象：特化 / 研钻
const YEAR2_TARGETS: [i32; 2] = [601, 701];

/// 第 3 年（turn 47）可用的融合对象：协助 / 不屈 / 兼修
const YEAR3_TARGETS: [i32; 3] = [801, 901, 1001];

/// `status_halve_factor` 的 serde 缺省值（0.5 = 超阈值减半）。
/// 不能直接写 `#[serde(default = "0.5")]`，故用函数。
fn default_halve_factor() -> f64 {
    0.5
}

/// `reserve_status_factor` 的 serde 缺省值（拉面 RESERVE_STATUS_FACTOR = 40）。
fn default_reserve_factor() -> f64 {
    40.0
}

/// 剧本专属配置段
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatsuriConfig {
    /// 融合（転成）预设
    pub fusion: FusionConfig,
    /// 手写训练员策略旋钮（S7；后续扫参）
    pub trainer: TrainerConfig,
}

/// 手写训练员策略旋钮（全部可扫参；默认档 = 攻略 / memo 方向）
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerConfig {
    /// 目标函数的 PT 偏好：`1.0` = 官方评分；引擎既有定档 `2.0`
    pub pt_favor_rate: f64,
    /// 体力低于此值进入「该休息了」区间（仍由打分表对比收益）
    pub rest_vital: i32,
    /// 発表会 / 本番达标保护的提前回合（0 = 关闭）
    pub checkpoint_margin: i32,
    /// Senior 合宿前把「有编成卡的演目」推到的最小 Lv（0 = 关闭）
    pub act_level_target: i32,
    /// 是否打自选比赛（false = 只在目标赛出手）
    pub self_race: bool,
    /// 友人外出是否尽早用完（攻略策略；false = 按收益打分）
    pub friend_outing_early: bool,
    /// 属性收益「减半」起算的原始属性值（0 = 关闭减半，完全按非减半表打分）。
    ///
    /// 口径：实机里某维属性超过该值后，单次训练/稽古的**实际入账**减半、到上限截断；
    /// 而本模拟器的 `five_status_final_score` 评分表是**非减半**的凸表（越高边际越大）。
    /// 决策器若直接按非减半表给「已接近上限的维」打满分，就会把回合浪费在吃不满的维上
    /// （复盘 T76：速度已贴上限仍选速度稽古，实机只入账个位数）。修正 = 决策打分时把
    /// 预览收益先按实机口径「上限截断 + 超阈值减半」折算成有效入账，再走非减半表换算回
    /// 评分点。阈值与上限都用模拟器的原始（非减半）坐标，无需再做面板换算。
    #[serde(default)]
    pub status_halve_threshold: i32,
    /// 超阈值部分保留的比例（0.5 = 减半；1.0 = 不减半；配合 `status_halve_threshold` 用）
    #[serde(default = "default_halve_factor")]
    pub status_halve_factor: f64,
    /// 属性溢出模式：false = 现行「上限截断 + 超阈值减半」；true = 拉面式「时间感知软上限储备」。
    /// 移植自 `xulai1001/umaai-rs` handwritten_evaluator.rs::status_soft_function（见
    /// lab/属性溢出软上限旋钮移植.md）。默认关闭，不改变现有行为。
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
    /// 友人外出截止回合（0 = 自动取 Senior 合宿窗口起点，即「第三年 10 月上」前后）。
    /// 截止前必须把 5 段用完，越接近截止、剩余段数越多，打分越急迫。
    #[serde(default)]
    pub friend_outing_deadline: i32,
    /// 友人外出截止急迫度权重（0 = 关闭急迫加分，只按段事件 + 演目收益打分）
    #[serde(default)]
    pub friend_outing_urgency: f64,
    /// 演目 Lv 质变卡阶奖励：把有编成卡的演目从 Lv7→8 / Lv8→9 推过质变点时的额外加分
    /// （技能pt 乘区 +80%→+100%→+135% 的大跳变；0 = 关闭）。吾妻外出抬最低演目 Lv 时同样吃到，
    /// 自动引导「外出卡质变点」（攻略优化点 4）。
    #[serde(default)]
    pub level_step_bonus: f64,
    /// 根性演目 Lv 封顶（0 = 关闭）：根 Lv 达到该值后，`level_value(根)` 归零——
    /// 「第二年末根强化到 5 就可以不用管了」。只影响推 Lv 的打分，演目槽仍照常累积。
    #[serde(default)]
    pub root_level_cap: i32,
    /// 根性训练最少人头（0 = 关闭）：根训练位担当数 < 该值时吃 `ROOT_TRAIN_PENALTY`
    /// 大额惩罚——「能不点根就不点根，除非 5 人头」。
    #[serde(default)]
    pub root_train_min_heads: usize,
    /// 友人外出 +Lv 是否避开根性演目（false = 沿用三则落点）——
    /// 「外出强化最好不要给根」。三则落点选中根时改挑非根演目里最低的；全是根才回落。
    #[serde(default)]
    pub outing_avoid_root: bool,
    /// 自选比赛解禁回合（0 = 关闭；仅 `self_race = true` 时有意义）：turn ≥ 该值才允许
    /// 把自选比赛纳入打分——「第三年天秋后要是训练烂可以跑 g1」（天秋 ≈ T65，取 66）。
    #[serde(default)]
    pub self_race_from_turn: i32,
    /// 五维策略权重（官方评分边际之外的偏好；默认全 1）
    pub status_weights: [f64; 5],
    /// 属性 / 资源 / 事件的折算权重
    pub weights: ValueWeights,
    /// 教诲分配价值权重
    pub advice: AdviceWeights,
    /// 动态权重（随回合 / 饱和度 / 进度调制上面三套静态权重）
    ///
    /// `#[serde(default)]`：默认层文件不写这一段也能解析，缺省即全 0 = 静态行为。
    #[serde(default)]
    pub dynamic: DynamicConfig,
}

/// 动态权重旋钮（逐回合调制；全部默认 `0.0` = 关闭，完全退化成静态打分）
///
/// 设计口径：官方评分表本身是凸的，贪心按边际打分已经能吃掉「越接近上限越值钱」，
/// 所以这里**不重复建模边际**，只补三件贪心看不见的事——
/// ① 五维之间的追赶（速先满就该把回合让给根 / 智）；
/// ② 演目 Lv 之间的追赶（治 `10/10/10/8/8` 这种长尾）；
/// ③ 体力与进度的时间价值（末期体力不值钱、领先就该卡回合、落后就该加速升级）。
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DynamicConfig {
    /// 五维追赶强度：`w[i] *= 1 + 2k(head[i] − mean_head)`，`head[i]` = 剩余空间占比
    pub status_catchup: f64,
    /// 演目 Lv 追赶强度：`level_value *= 1 + k·max(0, mean_lv − lv[act])`
    pub act_catchup: f64,
    /// 体力影子价格随回合衰减：`weights.vital *= 1 − k·(turn / max_turn)`
    pub vital_decay: f64,
    /// 满体力闲置惩罚放大：`idle_penalty *= 1 + k·(vital − 70) / 30`
    pub vital_idle: f64,
    /// 稽古节奏（卡回合 ↔ 加速升级）：`level_value *= 1 + k·(达标线 − 短板 Lv) / 3`
    pub keiko_tempo: f64,
}

impl Default for DynamicConfig {
    /// 全 0 = 关闭全部动态调制（与上游静态行为逐位相同）
    fn default() -> Self {
        Self {
            status_catchup: 0.0,
            act_catchup: 0.0,
            vital_decay: 0.0,
            vital_idle: 0.0,
            keiko_tempo: 0.0,
        }
    }
}

/// 属性 / 资源 / 事件的折算权重（单位 = 评分点）
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueWeights {
    /// 每点体力基础价值（再乘分段系数）
    pub vital: f64,
    /// 每级干劲
    pub motivation: f64,
    /// 每点羁绊
    pub friendship: f64,
    /// 每级灵感
    pub hint: f64,
    /// 每点演目槽
    pub gauge: f64,
    /// 每级演目 Lv（普通价值；不含达标急迫）
    pub act_level: f64,
    /// 発表会每缺 1 个演目达标 / 每缺 1 级的惩罚
    pub checkpoint: f64,
    /// 【变胖】状态的期望惩罚（一次性）
    pub fat: f64,
    /// 友人外出每段的基础价值（演目 Lv+1 与段事件之外的「尽早用完」策略分；
    /// 替代原先写死的 `FRIEND_OUTING_EARLY_BONUS`，可扫参）
    #[serde(default)]
    pub friend_outing: f64,
}

/// 教诲分配价值权重（特征系数）
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdviceWeights {
    /// 每张支援卡担当权重
    pub card: f64,
    /// 每名编成外神驹担当权重
    pub npc: f64,
    /// 预计普通训练次数权重
    pub train: f64,
    /// 预计演目稽古次数权重
    pub keiko: f64,
    /// 绑定演目 Lv 权重
    pub level: f64,
    /// 発表会达标急迫度权重
    pub milestone: f64,
    /// 特化上限（601 特別稽古的 `status_limit`）权重
    ///
    /// `#[serde(default)]`：旧覆盖层不写也能解析，缺省 `0.0` = 与既有行为逐位一致。
    #[serde(default)]
    pub status_limit: f64,
    /// 助力の支え（401 的 `support_assist` 开关量）权重
    #[serde(default)]
    pub assist: f64,
    /// 落位先验权重：命中一位离线搜出的落位加这么多分（0 = 关闭）
    ///
    /// 先验表见 [`crate::value::ROUTE_PRIOR`]，由 `scripts/tabu_route.py` 离线搜出后内化，
    /// 目的是不再依赖 `MATSURI_ASSIGN_PIN` 外部钉位。取值足够大时等价于硬钉位。
    #[serde(default)]
    pub route: f64,
}

/// 融合预设：第 2 年 2 组、第 3 年剩余 3 组
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionConfig {
    /// 第 2 年（turn 23 结束）：恰好 2 组
    pub year2: Vec<FusionPair>,
    /// 第 3 年（turn 47 结束）：剩余 3 组
    pub year3: Vec<FusionPair>,
}

/// 一组融合配对（基础教诲 → 融合对象）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionPair {
    /// 被融合的基础教诲 id（101–501）
    pub base: i32,
    /// 融合对象 id（第 2 年 601 / 701，第 3 年 801 / 901 / 1001）
    pub target: i32,
}

/// 覆盖层：与默认层同构、字段全 `Option`（`None` = 沿用默认层）
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatsuriConfigOverlay {
    fusion: Option<FusionOverlay>,
    /// 训练员段：给了就**整段替换**（不按字段合并，避免半套旋钮交叉污染）
    trainer: Option<TrainerConfig>,
}

/// `fusion` 段的覆盖层（整段替换；配对列表不按元素合并）
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionOverlay {
    year2: Option<Vec<FusionPair>>,
    year3: Option<Vec<FusionPair>>,
}

impl OverlayConfig for MatsuriConfig {
    type Overlay = MatsuriConfigOverlay;

    fn apply_overlay(&mut self, overlay: Self::Overlay) {
        if let Some(fusion) = overlay.fusion {
            if let Some(year2) = fusion.year2 {
                self.fusion.year2 = year2;
            }
            if let Some(year3) = fusion.year3 {
                self.fusion.year3 = year3;
            }
        }
        if let Some(trainer) = overlay.trainer {
            self.trainer = trainer;
        }
    }

    fn validate(&self) -> Result<()> {
        MatsuriConfig::validate(self)
    }
}

impl MatsuriConfig {
    /// 解析 TOML 文本（默认层 + 可选覆盖层；测试 / 内嵌场景复用）
    pub fn parse(base_toml: &str, overlay_toml: Option<&str>) -> Result<Self> {
        parse_with_overlay(base_toml, overlay_toml)
    }

    /// 载入：默认层文件 + 显式指定的覆盖层文件（给了就必须存在）
    pub fn load(base: impl AsRef<Path>, overlay: Option<&Path>) -> Result<Self> {
        let base = base.as_ref();
        let base_text = std::fs::read_to_string(base)
            .with_context(|| format!("读取剧本配置失败: {}", base.display()))?;
        let overlay_text = match overlay {
            Some(path) => Some(
                std::fs::read_to_string(path)
                    .with_context(|| format!("读取剧本覆盖配置失败: {}", path.display()))?,
            ),
            None => None,
        };
        Self::parse(&base_text, overlay_text.as_deref())
            .with_context(|| format!("剧本配置来源: {}", base.display()))
    }

    /// 载入：默认层 + **约定覆盖层**（`<dir>/matsuri_config.toml`，存在才合并）
    ///
    /// 与引擎 `Config::load_with_overlay_file` 同语义：覆盖层缺席不报错。
    pub fn load_with_overlay_file(
        base_dir: impl AsRef<Path>,
        dir: impl AsRef<Path>,
    ) -> Result<Self> {
        let overlay = dir.as_ref().join(MATSURI_CONFIG_FILE);
        let overlay = overlay.exists().then_some(overlay);
        Self::load(
            base_dir.as_ref().join(MATSURI_CONFIG_FILE),
            overlay.as_deref(),
        )
    }

    /// 载入期校验：第 2 年恰 2 组、第 3 年恰 3 组；base 只用 101–501 且不重复；
    /// target 年份内各自恰好用一次
    fn validate(&self) -> Result<()> {
        let fusion = &self.fusion;
        ensure!(
            fusion.year2.len() == YEAR2_TARGETS.len(),
            "fusion.year2 应恰好 {} 组，实际 {}",
            YEAR2_TARGETS.len(),
            fusion.year2.len()
        );
        ensure!(
            fusion.year3.len() == YEAR3_TARGETS.len(),
            "fusion.year3 应恰好 {} 组，实际 {}",
            YEAR3_TARGETS.len(),
            fusion.year3.len()
        );
        let mut bases = Vec::new();
        for (i, pair) in fusion.year2.iter().enumerate() {
            ensure!(
                BASE_ADVICE_IDS.contains(&pair.base),
                "fusion.year2[{i}].base 越界: {}（应为 101–501）",
                pair.base
            );
            ensure!(
                YEAR2_TARGETS.contains(&pair.target),
                "fusion.year2[{i}].target 越界: {}（应为 601 / 701）",
                pair.target
            );
            ensure!(
                !bases.contains(&pair.base),
                "fusion.year2[{i}].base 重复: {}",
                pair.base
            );
            bases.push(pair.base);
        }
        for (i, pair) in fusion.year3.iter().enumerate() {
            ensure!(
                BASE_ADVICE_IDS.contains(&pair.base),
                "fusion.year3[{i}].base 越界: {}（应为 101–501）",
                pair.base
            );
            ensure!(
                YEAR3_TARGETS.contains(&pair.target),
                "fusion.year3[{i}].target 越界: {}（应为 801 / 901 / 1001）",
                pair.target
            );
            ensure!(
                !bases.contains(&pair.base),
                "fusion.year3[{i}].base 重复: {}（第 3 年只用剩余基础教诲）",
                pair.base
            );
            bases.push(pair.base);
        }
        ensure!(
            bases.len() == BASE_ADVICE_IDS.len(),
            "fusion 应覆盖全部 {} 个基础教诲，实际 {} 个",
            BASE_ADVICE_IDS.len(),
            bases.len()
        );
        let mut year2_targets: Vec<i32> = fusion.year2.iter().map(|pair| pair.target).collect();
        year2_targets.sort_unstable();
        ensure!(
            year2_targets == YEAR2_TARGETS,
            "fusion.year2 的 target 应各自恰好使用一次"
        );
        let mut year3_targets: Vec<i32> = fusion.year3.iter().map(|pair| pair.target).collect();
        year3_targets.sort_unstable();
        ensure!(
            year3_targets == YEAR3_TARGETS,
            "fusion.year3 的 target 应各自恰好使用一次"
        );
        validate_trainer(&self.trainer)
    }
}

/// 训练员旋钮校验：范围 + 非负 / 有限（扫参硬边界；非法值直接拒绝，不静默回退）
fn validate_trainer(trainer: &TrainerConfig) -> Result<()> {
    ensure!(
        trainer.pt_favor_rate.is_finite() && trainer.pt_favor_rate > 0.0,
        "trainer.pt_favor_rate 应为正数"
    );
    ensure!(
        (0..=100).contains(&trainer.rest_vital),
        "trainer.rest_vital 应在 0–100"
    );
    ensure!(
        (0..=20).contains(&trainer.checkpoint_margin),
        "trainer.checkpoint_margin 应在 0–20"
    );
    ensure!(
        (0..=10).contains(&trainer.act_level_target),
        "trainer.act_level_target 应在 0–10"
    );
    for (index, weight) in trainer.status_weights.iter().enumerate() {
        ensure!(
            weight.is_finite() && *weight >= 0.0,
            "trainer.status_weights[{index}] 应为非负有限数"
        );
    }
    let weights = &trainer.weights;
    let entries: [(&str, f64); 9] = [
        ("weights.vital", weights.vital),
        ("weights.motivation", weights.motivation),
        ("weights.friendship", weights.friendship),
        ("weights.hint", weights.hint),
        ("weights.gauge", weights.gauge),
        ("weights.act_level", weights.act_level),
        ("weights.checkpoint", weights.checkpoint),
        ("weights.fat", weights.fat),
        ("weights.friend_outing", weights.friend_outing),
    ];
    for (name, value) in entries {
        ensure!(
            value.is_finite() && value >= 0.0,
            "trainer.{name} 应为非负有限数"
        );
    }
    ensure!(
        (0..=4000).contains(&trainer.status_halve_threshold),
        "trainer.status_halve_threshold 应在 0–4000（0 = 关闭减半）"
    );
    ensure!(
        trainer.status_halve_factor.is_finite() && (0.0..=1.0).contains(&trainer.status_halve_factor),
        "trainer.status_halve_factor 应在 0–1（0.5 = 超阈值减半）"
    );
    ensure!(
        trainer.reserve_status_factor.is_finite()
            && (0.0..=200.0).contains(&trainer.reserve_status_factor),
        "trainer.reserve_status_factor 应在 0–200（0 = 关闭软储备）"
    );
    ensure!(
        (0..=600).contains(&trainer.final_bonus),
        "trainer.final_bonus 应在 0–600"
    );
    ensure!(
        (0..=77).contains(&trainer.friend_outing_deadline),
        "trainer.friend_outing_deadline 应在 0–77（0 = 自动取 Senior 合宿起点）"
    );
    ensure!(
        trainer.friend_outing_urgency.is_finite() && trainer.friend_outing_urgency >= 0.0,
        "trainer.friend_outing_urgency 应为非负有限数"
    );
    ensure!(
        trainer.level_step_bonus.is_finite() && (0.0..=2000.0).contains(&trainer.level_step_bonus),
        "trainer.level_step_bonus 应在 0–2000（0 = 关闭质变卡阶奖励）"
    );
    ensure!(
        (0..=10).contains(&trainer.root_level_cap),
        "trainer.root_level_cap 应在 0–10（0 = 关闭根 Lv 封顶）"
    );
    ensure!(
        trainer.root_train_min_heads <= 5,
        "trainer.root_train_min_heads 应在 0–5（0 = 关闭根训练人头门槛）"
    );
    ensure!(
        (0..=77).contains(&trainer.self_race_from_turn),
        "trainer.self_race_from_turn 应在 0–77（0 = 关闭自选赛回合门槛）"
    );
    let advice = &trainer.advice;
    let entries: [(&str, f64); 9] = [
        ("advice.card", advice.card),
        ("advice.npc", advice.npc),
        ("advice.train", advice.train),
        ("advice.keiko", advice.keiko),
        ("advice.level", advice.level),
        ("advice.milestone", advice.milestone),
        ("advice.status_limit", advice.status_limit),
        ("advice.assist", advice.assist),
        ("advice.route", advice.route),
    ];
    for (name, value) in entries {
        ensure!(
            value.is_finite() && value >= 0.0,
            "trainer.{name} 应为非负有限数"
        );
    }
    let dynamic = &trainer.dynamic;
    let entries: [(&str, f64, f64); 5] = [
        ("dynamic.status_catchup", dynamic.status_catchup, 4.0),
        ("dynamic.act_catchup", dynamic.act_catchup, 2.0),
        ("dynamic.vital_decay", dynamic.vital_decay, 1.0),
        ("dynamic.vital_idle", dynamic.vital_idle, 4.0),
        ("dynamic.keiko_tempo", dynamic.keiko_tempo, 4.0),
    ];
    for (name, value, upper) in entries {
        ensure!(
            value.is_finite() && (0.0..=upper).contains(&value),
            "trainer.{name} 应在 0–{upper}（实际 {value}）"
        );
    }
    Ok(())
}

impl Default for MatsuriConfig {
    /// 与仓内默认层 `data/matsuri_config.toml` 同值（未注入配置的宿主 / 测试兜底）
    fn default() -> Self {
        Self {
            fusion: FusionConfig {
                year2: vec![
                    FusionPair {
                        base: 101,
                        target: 601,
                    },
                    FusionPair {
                        base: 201,
                        target: 701,
                    },
                ],
                year3: vec![
                    FusionPair {
                        base: 401,
                        target: 801,
                    },
                    FusionPair {
                        base: 301,
                        target: 901,
                    },
                    FusionPair {
                        base: 501,
                        target: 1001,
                    },
                ],
            },
            trainer: TrainerConfig::default(),
        }
    }
}

impl Default for TrainerConfig {
    /// 与仓内默认层 `[trainer]` 段同值（攻略 / memo 方向；后续扫参）
    fn default() -> Self {
        Self {
            pt_favor_rate: 2.0,
            rest_vital: 45,
            checkpoint_margin: 3,
            act_level_target: 8,
            self_race: false,
            friend_outing_early: true,
            status_halve_threshold: 0,
            status_halve_factor: 0.5,
            status_soft_reserve: false,
            reserve_status_factor: 40.0,
            final_bonus: 0,
            friend_outing_deadline: 0,
            friend_outing_urgency: 0.0,
            level_step_bonus: 0.0,
            root_level_cap: 0,
            root_train_min_heads: 0,
            outing_avoid_root: false,
            self_race_from_turn: 0,
            status_weights: [1.0; 5],
            weights: ValueWeights {
                vital: 1.2,
                motivation: 30.0,
                friendship: 0.6,
                hint: 1.0,
                gauge: 25.0,
                act_level: 120.0,
                checkpoint: 800.0,
                fat: 120.0,
                friend_outing: 120.0,
            },
            advice: AdviceWeights {
                card: 1.0,
                npc: 0.6,
                train: 1.0,
                keiko: 1.0,
                level: 1.0,
                milestone: 1.0,
                status_limit: 0.0,
                assist: 0.0,
                route: 0.0,
            },
            dynamic: DynamicConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 仓内默认层文本
    fn repo_text() -> String {
        std::fs::read_to_string(crate::testing::data_dir().join(MATSURI_CONFIG_FILE))
            .expect("仓内剧本配置默认层应存在")
    }

    #[test]
    fn parse_repo_default() {
        let cfg = MatsuriConfig::parse(&repo_text(), None).expect("默认层应可解析");
        assert_eq!(cfg, MatsuriConfig::default());
    }

    #[test]
    fn unknown_key_rejected() {
        let bad = repo_text().replace("year2", "year_two");
        let err = MatsuriConfig::parse(&bad, None).unwrap_err();
        assert!(err.to_string().contains("解析默认配置失败"));
    }

    #[test]
    fn overlay_replaces_year2_only() {
        let overlay =
            "[fusion]\nyear2 = [{ base = 201, target = 601 }, { base = 101, target = 701 }]\n";
        let cfg = MatsuriConfig::parse(&repo_text(), Some(overlay)).expect("覆盖层应可合并");
        assert_eq!(
            cfg.fusion.year2,
            vec![
                FusionPair {
                    base: 201,
                    target: 601
                },
                FusionPair {
                    base: 101,
                    target: 701
                },
            ]
        );
        assert_eq!(cfg.fusion.year3, MatsuriConfig::default().fusion.year3);
    }

    #[test]
    fn duplicate_base_rejected() {
        let bad = repo_text().replace("base = 201\ntarget = 701", "base = 101\ntarget = 701");
        assert!(
            MatsuriConfig::parse(&bad, None).is_err(),
            "基础教诲重复应报错"
        );
    }

    #[test]
    fn wrong_target_rejected() {
        let bad = repo_text().replace("target = 601", "target = 801");
        assert!(
            MatsuriConfig::parse(&bad, None).is_err(),
            "第 2 年 target 越界应报错"
        );
    }

    #[test]
    fn wrong_count_rejected() {
        let bad = repo_text().replace("[[fusion.year2]]\nbase = 201\ntarget = 701\n\n", "");
        assert!(
            MatsuriConfig::parse(&bad, None).is_err(),
            "第 2 年组数不对应报错"
        );
    }

    #[test]
    fn trainer_overlay_requires_full_section() {
        let overlay = "[trainer]\npt_favor_rate = 1.0\n";
        assert!(
            MatsuriConfig::parse(&repo_text(), Some(overlay)).is_err(),
            "trainer 段覆盖必须整段给出（不按字段合并）"
        );
    }

    #[test]
    fn trainer_overlay_replaces_whole_section() {
        let overlay = "[trainer]\n\
                       pt_favor_rate = 1.0\n\
                       rest_vital = 40\n\
                       checkpoint_margin = 2\n\
                       act_level_target = 0\n\
                       self_race = true\n\
                       friend_outing_early = false\n\
                       status_weights = [2.0, 1.0, 1.0, 1.0, 1.0]\n\
                       [trainer.weights]\n\
                       vital = 1.0\n\
                       motivation = 20.0\n\
                       friendship = 0.5\n\
                       hint = 1.0\n\
                       gauge = 10.0\n\
                       act_level = 50.0\n\
                       checkpoint = 100.0\n\
                       fat = 10.0\n\
                       [trainer.advice]\n\
                       card = 1.0\n\
                       npc = 1.0\n\
                       train = 1.0\n\
                       keiko = 1.0\n\
                       level = 1.0\n\
                       milestone = 1.0\n";
        let cfg = MatsuriConfig::parse(&repo_text(), Some(overlay)).expect("整段覆盖应可合并");
        assert_eq!(cfg.trainer.pt_favor_rate, 1.0);
        assert_eq!(cfg.trainer.rest_vital, 40);
        assert!(cfg.trainer.self_race);
        assert_eq!(
            cfg.fusion,
            MatsuriConfig::default().fusion,
            "fusion 段不受影响"
        );
        assert_eq!(
            cfg.trainer.dynamic,
            DynamicConfig::default(),
            "trainer.dynamic 缺省 = 全 0（关闭动态调制，退化成静态打分）"
        );
    }

    /// 动态权重旋钮越界应被拒绝（扫参硬边界，不静默回退）
    #[test]
    fn dynamic_bad_values_rejected() {
        let overlay = "[trainer]\n\
                       pt_favor_rate = 2.0\n\
                       rest_vital = 45\n\
                       checkpoint_margin = 3\n\
                       act_level_target = 8\n\
                       self_race = false\n\
                       friend_outing_early = true\n\
                       status_weights = [1.0, 1.0, 1.0, 1.0, 1.0]\n\
                       [trainer.weights]\n\
                       vital = 1.2\n\
                       motivation = 30.0\n\
                       friendship = 0.6\n\
                       hint = 1.0\n\
                       gauge = 25.0\n\
                       act_level = 120.0\n\
                       checkpoint = 800.0\n\
                       fat = 120.0\n\
                       [trainer.advice]\n\
                       card = 1.0\n\
                       npc = 0.6\n\
                       train = 1.0\n\
                       keiko = 1.0\n\
                       level = 1.0\n\
                       milestone = 1.0\n\
                       [trainer.dynamic]\n\
                       vital_decay = 1.5\n";
        assert!(
            MatsuriConfig::parse(&repo_text(), Some(overlay)).is_err(),
            "vital_decay 超过 1 应报错"
        );
        let bad = overlay.replace("vital_decay = 1.5", "status_catchup = -0.5");
        assert!(
            MatsuriConfig::parse(&repo_text(), Some(&bad)).is_err(),
            "负的 status_catchup 应报错"
        );
        let ok = overlay.replace("vital_decay = 1.5", "keiko_tempo = 2.0");
        let cfg = MatsuriConfig::parse(&repo_text(), Some(&ok)).expect("合法动态档应可解析");
        assert_eq!(cfg.trainer.dynamic.keiko_tempo, 2.0);
        assert_eq!(cfg.trainer.dynamic.act_catchup, 0.0, "未写的旋钮取 0");
    }

    #[test]
    fn trainer_bad_values_rejected() {
        let bad = repo_text().replace("rest_vital = 45", "rest_vital = -1");
        assert!(
            MatsuriConfig::parse(&bad, None).is_err(),
            "体力阈值越界应报错"
        );
        let bad = repo_text().replace("pt_favor_rate = 2.0", "pt_favor_rate = 0.0");
        assert!(
            MatsuriConfig::parse(&bad, None).is_err(),
            "PT 偏好非正数应报错"
        );
        let bad = repo_text().replace("vital = 1.2", "vital = -0.1");
        assert!(MatsuriConfig::parse(&bad, None).is_err(), "权重为负应报错");
    }
}
