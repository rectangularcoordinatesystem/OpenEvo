# OpenEvo × DiscoveryBench 进化实验阶段总结

## 一、实验目标

在 DiscoveryBench validation 上运行 OpenEvo 进化，汇总并冻结最终策略，再在完整 test 上评估进化后的泛化表现。

官方数据划分为 25 条 validation 和 239 条 test。正确流程是：

**25 条 validation 进化 → 生成 Global Policy → 冻结策略 → 239 条 test 评测**

## 二、已完成实验与结果

### 1. 25 条 Validation 进化

每个样本最多运行 evo 0–3，结果如下：

| 指标 | 进化前 | 进化后 | 最佳 |
|---|---:|---:|---:|
| 平均分 | 0.330 | 0.450 | 0.476 |
| 满分率 | 12% | 24% | - |

25 条中，9 条最终提升、2 条退步、14 条不变。平均分绝对提升 0.120，约提升 36.4%，说明 sample 内部的 retry evolution 确实产生了正向效果。

```text
平均分
进化前  0.330  █████████████
进化后  0.450  ██████████████████
最佳值  0.476  ███████████████████
```

### 2. 两次未完成的 Frozen Test

两次 test 均正确加载了 validation 生成的同一个 Global Policy，并采用 5 sample / 5 sandbox 并发，但都被中途暂停。

| 运行 | 已完成 | 部分均分 | 工具错误率 |
|---|---:|---:|---:|
| Test A | 99/239 | 0.085 | 96.5% |
| Test B | 54/239 | 0.113 | 96.7% |

由于两次 test 均未完成，且并发完成样本的领域分布不一致，这些部分均分不能作为最终 test 成绩，也不能直接与 validation 均分比较。

## 三、发现的主要问题

### 1. Validation 样本数量识别错误

早期曾将 validation 按 239 条运行，但其实际只有 25 条。第 26 条以后 Inspect 未选中样本，adapter 却将空运行记录为 `success + 0 分`，并继续生成 memory/skill 等文件。

现已修复：validation 固定识别为 25 条、test 为 239 条；空 sample 会标记为 `no_samples` 并立即停止。

### 2. Python 工具调用大面积失效

旧 validation 的 1189 个工具结果中有 1055 个失败，错误率 88.7%；两次 test 的错误率进一步达到约 96%。错误完全一致：

```text
python_session {}
'code' is a required property
```

根因是 solver 只把工具的 `content` 返回给模型，却遗漏了独立的 `error` 字段。模型不知道参数错误，因此反复提交空参数，直至耗尽 24 个步骤。

该问题与 `ASTA_TOOL_KEY` 无关。`python_session` 是 Docker/Jupyter 本地工具，错误发生在进入 sandbox 之前。

### 3. 旧 Global Policy 汇总不完整

旧 promotion 只读取最前面的 20 个失败 attempt，没有系统利用每个 validation 样本的最佳轨迹和成功经验，导致最终 1.4 KB 的 Global Policy 偏向“数据不足时保守回答”。当 Python 工具失败时，这一策略进一步诱导模型输出“无法确定”，从而大量获得 0 分。

## 四、已完成的代码修改

- 修复 validation/test 样本数识别及空评测保护；
- 明确要求 `python_session` 必须传入非空 `code`；
- 将工具 `error` 完整反馈给模型；
- 工具参数失败不再消耗有效分析步骤，并允许有限纠错；
- Global Policy 改为汇总全部 validation 样本的最佳轨迹；
- 记录最终回退样本，避免只学习失败或最后一轮；
- 新增相应单元测试，完整测试结果为 **28 passed**。

## 五、修复验证

修复后进行了 1 条 validation smoke test：

| 指标 | 修复前典型情况 | 修复后 |
|---|---:|---:|
| 缺少 `code` 错误 | 可连续接近 24 次 | 1 次 |
| 有效工具结果 | 很少或为 0 | 10 次 |
| 工具错误率 | 88%–97% | 9.1% |

模型在首次参数错误后成功纠正并继续执行，说明工具反馈链路已经恢复。

## 六、当前结论与下一步

目前可以确认：validation 内部进化存在提升，但旧 test 的低分主要由工具执行故障和不完整的 Global Policy 汇总造成，不能据此判断 OpenEvo 本身进化无效。

下一步计划：

1. 使用修复后的 adapter 在新目录重跑完整 25 条 validation；
2. 检查工具错误率并审查新版 Global Policy；
3. 完整运行 239 条无 Policy frozen baseline；
4. 完整运行 239 条注入 Policy 的 frozen test；
5. 按相同 sample ID 配对比较分数、提升/退步数量及不同领域表现。

只有完成 baseline 与 evolved test 的同配置配对实验后，才能对 OpenEvo 的最终泛化提升作出可信判断。
