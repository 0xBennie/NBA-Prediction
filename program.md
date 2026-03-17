# NBA Prediction Market — 自学习研究计划

> 受 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 启发：
> 用自主实验循环不断优化参数，提高预测胜率。

## 设计哲学

autoresearch: 改 `train.py` → 跑5分钟训练 → 看 `val_bpb` → keep/discard
我们:        改 `scoring_params.json` → 回测历史信号 → 看 Sharpe-like metric → keep/discard

## 自学习循环

```
LOOP (每次有新结果回填时触发):
  1. 加载当前 scoring_params.json 作为 baseline
  2. 计算 baseline 的综合指标 (Sharpe-like metric)
  3. 随机变异1-2个参数维度 (±10-20% 扰动)
  4. 用 signal_log 历史数据回测新参数
  5. IF metric 提升 → 保存新参数 (advance)
  6. IF metric 没提升 → 回滚 (revert)
  7. 记录实验到 experiments.tsv
  8. 重复 5 次 (每轮)
```

## 可变异的参数维度

| 维度 | 当前值 | 范围 | 影响 |
|------|--------|------|------|
| edge_max | 40 | 20-55 | Edge评分在总分中的权重 |
| line_movement_max | 25 | 10-35 | 盘口移动权重 |
| injury_max | 20 | 5-30 | 伤病权重 |
| price_position_max | 15 | 5-25 | 价格位置权重 |
| noise_floor | 3% | 2-5% | Edge超过此值视为噪声 |
| sweet_min | 0.8% | 0.4-1.5% | Edge甜区下限 |
| game_threshold | 50 | 35-65 | 推送阈值 |
| min_raw_edge | 0.4% | 0.2-0.8% | 最低edge要求 |
| kelly.fraction | 1/4 | 1/10-1/2 | Kelly仓位比例 |

## 评估指标

```python
metric = sharpe + win_rate * 0.5 + selectivity_bonus + total_pnl * 10
```

- **sharpe**: mean(kelly_weighted_roi) / std(kelly_weighted_roi)
- **win_rate**: 推送信号胜率
- **selectivity_bonus**: 推送率偏离15%目标的惩罚
- **total_pnl**: Kelly加权总盈亏

## 约束

- 参数变异在预设范围内（不会产生荒谬值）
- 至少30条已结算信号才开始学习
- 每次只改1-2个维度（控制变量法）
- 所有实验都记录到 experiments.tsv（可审计）

## 文件结构

```
scoring_params.json  — 当前最优参数（自动更新）
experiments.tsv      — 实验记录（类似autoresearch的results.tsv）
ml/auto_learner.py   — 自学习引擎
program.md           — 本文件（研究计划）
```

## CLI 命令

```bash
python main.py --learn          # 手动跑一轮学习（20次实验）
python main.py --learn-report   # 查看学习进度
python main.py                  # 持续模式（每次回填后自动学习）
```
