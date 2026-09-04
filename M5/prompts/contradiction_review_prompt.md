# M5 长程明确矛盾重评协议

## 1. 目的与盲评

本轮只重评六部小说的“矛盾/万字”。评审对象使用匿名编号 E01–E06；评审不得读取
`evaluation/controller/`、旧 `evaluation/results/`、方法提示词、运行日志或旧汇总表。
六部小说使用相同模型 `gpt-5.6-sol`、相同 `medium` 推理强度和本协议。

## 2. 冻结审查范围

每个匿名包只允许读取：

- `review_manifest.json`：冻结的钩子事实与谓词编号；
- `hook_chapter_index.json`：每条钩子的固定章节集合；
- `novel/chapter_*.md`：上述索引实际列出的正文。

审查全部10条钩子轨迹。分母是索引中所有章节去重后的汉字数；同一章被多条钩子引用时
只计一次。该指标命名为“固定钩子轨迹长程明确矛盾/万字”，不得称为全书穷尽计数。

## 3. 唯一计数单位

一个计数单位是一组唯一的 `(hook_id, entity, attribute)` 冲突，保存为 `conflict_key`。
同一属性在多个后续章节重复漂移只计一次，并可在一个条目中增加辅助锚点。若一个漂移同时
违反 seed 与 state 谓词，只建一个条目，在 `predicate_ids` 中列出全部对应编号。

优先关联最具体的 seed 谓词；只有 seed 未表达该状态关系时才单独使用 state 谓词。

## 4. 计入条件

只有同时满足以下全部条件才记为 `explicit_conflict`：

1. 源锚点和目标锚点均为小说正文中的实际断言；
2. 两者指向同一规范化实体和同一属性；
3. 两个属性值在重叠有效时间内不能同时成立；
4. 目标章晚于源章，且两章均属于该钩子的冻结章节索引；
5. 中间没有正文明确给出的、因果合理的状态改变或纠错；
6. 两段引文均为正文逐字子串，每段尽量不超过100个汉字；
7. 冲突能关联至少一个该钩子的 seed/state 谓词；
8. 置信度不低于0.80。

人物、物品、时间、空间、世界规则、因果分别记为 `person`、`item`、`time`、`space`、
`world_rule`、`causal`。

## 5. 不计入条件

以下情况一律不计：

- 后文未提及、遗忘、换用其他方案或钩子没有兑现；
- 疑问、假设、梦境、比喻、未经证实的角色猜测；
- 角色撒谎或错误记录，且叙事随后明确纠正；
- 有明确原因的转移、修复、损耗、升级、改造或时间变化；
- 只与金标预期冲突、但不与本小说已经写出的早期事实冲突；
- 两个不同实体、不同时间切片或不同版本之间的表面差异；
- 无法提供两处逐字锚点的印象判断。

边界候选写入 `excluded_candidates`，并给出排除原因，不进入计数。

## 6. 结果格式与派生公式

每位评审只提交原子条目，不手写矛盾总数或密度。聚合器校验匿名编号、章节范围、逐字引文、
谓词编号和 `conflict_key` 唯一性后派生：

`count = 通过校验的唯一 conflict_key 数`

`density_per_10000 = count / reviewed_unique_han_chars * 10000`

结果同时报告 `reviewed_unique_chapters`、`reviewed_unique_han_chars`、分类计数和排除候选数。

## 7. 输出要求

结果写到 `contradiction_reaudit/results/<EID>/result.json`。不得修改小说、旧评价结果、公共协议
或其他评审的文件。每个冲突条目必须包括：

- `conflict_key`, `hook_id`, `predicate_ids`, `category`；
- `entity`, `attribute`, `source_value`, `target_value`；
- `source` 与 `target` 的 `chapter_id`、逐字 `quote`、`assertion_status`；
- `conflict_reason`, `intervening_explanation_checked`, `confidence`。
