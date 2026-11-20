# 论文实验完成版（2026-11-20）

本目录是基于全部 60 个任务完成结果生成的最新英文审阅稿。原 2026-11-16 版本保留，不覆盖历史记录。

## 阅读与编辑

- [PDF 论文](manuscript.pdf)
- [Word 可编辑论文](manuscript.docx)
- [Markdown 源稿](manuscript.md)
- [HTML 预览](manuscript.html)

当前为单栏审阅排版，尚未套用最终投稿模板。

## 本次更新

1. 表 I 全部替换为本轮 P1 五种子测试结果，包括 CNN、固定尺度及身份增强的 GCN/HGT、完整 QC-HGT。完整模型 Clean Macro-F1 为 **0.7542 ± 0.0276**，Avg. Robustness 为 **0.7161 ± 0.0254**。
2. 表 II 全部替换为本轮 P2 结果，并新增同一 QC 骨干“关闭两个模块”对照。完整模型 Clean/Seen/Unseen Macro-F1 为 **0.7804 ± 0.0112 / 0.7729 ± 0.0143 / 0.7748 ± 0.0108**。
3. 保留验证集输入消融表 III，新增独立测试输入对照表 IV、测试 Accuracy/FAR/ECE/NLL 表 V、P1 参数与训练耗时表 VI。
4. 同步重写摘要、引言贡献、实验分析与结论；删除“尚待测试”等过时描述。新增噪声/漂移强度测试曲线图 3。
5. 明确主要提升来自输入表示改进；完整模块组合对同骨干无模块的额外 Macro-F1 均值增益约为 **1.34–1.39 个百分点**。单独启用任一模块均未提高无模块对照的均值。不声称统计显著或独立模块必然有效。
6. 保留重要限制：P2 Clean Accuracy 为 **71.44%**，不是 Macro-F1 的 78.04%；正常 FAR 仍高；极端噪声/漂移下排名可能反转；训练成本较高。
7. 当前没有新增近期外部模型实验，故未插入其文献成绩或声称达到 SOTA。

## 数据与复核

- 原始结果：`../../results/paper_revision_20261115/`，共 60 份完整测试结果。
- [表格完整精度与结果 SHA256](verified_tables.json)
- [表格 CSV](verified_tables.csv)
- [独立复核结果](independent_audit.json)
- 五种子均为 seed 0–4；所有 ± 使用样本标准差（ddof=1）。
- 表 I 先在每个 seed 内聚合强度，再跨 seed 汇总。Missing 合并随机/连续缺失共八档；Avg. Robustness 对五种退化子类型等权，共二十个强度条件，排除 Clean。
- 已核对新训练结果、manifest 和最佳权重/测试权重 SHA；比较组训练与测试选窗及尺度记录一致。CNN 实际保留窗口内标准化，正文披露该差异。
- 图 1、图 2 仍是验证集分析；图 3 与主表来自独立测试集。

## 再生成

在项目根目录执行：

```bash
XDG_CACHE_HOME=/tmp/qchgt-cache MPLCONFIGDIR=/tmp/qchgt_mpl .venv/bin/python papers/revision_20261120/update_manuscript.py
.venv/bin/python papers/revision_20261120/render_manuscript.py
```

第一条会依据原始结果重新生成正文、表格与图 3，并覆盖本目录 Markdown 稿，因此手动编辑正文后应同步修改生成脚本或只运行第二条转换命令。第二条使用 LibreOffice 生成含嵌入图片的 Word 和 PDF。
