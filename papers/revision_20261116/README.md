# 论文修订稿（2026-11-16）

论文：Graph Learning with Quality and Consistency Awareness for Safety Risk Identification。

本文件夹保存2026年11月16日形成的英文审阅稿快照。补充实验仍在运行；本稿不包含后续产生的新测试成绩，也不是最终投稿排版版本。

## 文件

- [PDF 审阅稿](manuscript.pdf)
- [Word 可编辑稿](manuscript.docx)：已嵌入两张图片。
- [Markdown 源稿](manuscript.md)
- [HTML 预览稿](manuscript.html)：与两张 PNG 图片放在同一目录。
- `input_validation.png`：五种子 A/B/C/D 验证 Macro-F1。
- `class_validation.png`：A 与 C 的逐类别验证 F1。

## 本次修订

1. 根据仓库五种子审计记录修正历史表 II，并调整摘要和结论中的模块排名表述。
2. 恢复历史 P1 CNN 基线，说明原始 QC-HGT 并非优于全部比较模型。
3. 加入 A/B/C/D 五种子验证结果、参数量、训练耗时、正常类误报及概率质量分析。
4. 补充训练集固定尺度、变量身份及工艺单元身份的输入方法描述。
5. 明确区分历史测试、本轮验证和待完成的新测试。C 的 77.40% 是验证 Macro-F1，不能当作未知组合测试成绩。

待补充实验完成并复核后，再整合新测试表、同骨干消融结果和最终结论。

历史结果依据：[五种子表格](../../docs/paper_facts_20261013_200609/tables.md)、[审计记录](../../docs/qc_hgt_audit_metrics_20261104.json)。
