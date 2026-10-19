# 本次变更

全部新增文件仅在独立 final_release 内；旧目录保护哈希全部一致。

- v2/des.py：A2原样副本，字节相同；未改物理语义。
- v2/milp.py：原Full纯整数路径增加完整上下界/行校验；无连续自由度时拒绝非法取整。
- queue_milp.py、recovery.py：固定整数连续LP恢复，完整验证与证据保存；原目标、候选、容量、deadline与1e-5阈值不变。
- spatial.py：主Spatial调用同一修复后queue求解器；保留原控制/派发逻辑。
- policies.py：原E嵌套控制器副本；所有分支使用新恢复函数。
- experiments.py、configs/protocol.json：实际fresh120核心与F/H重跑、逐请求/逐事件/账本和恢复次数；独立种子与共同随机条件不改。
- tests/、run_tests.py：已有70测试只读执行，加13数值/保护测试；初始失败尝试保留。
- analyze_release.py、validate_release.py：种子级配对统计、旧新差异、独立逐小时碳积分、物理容量/能耗/请求守恒与重复验证。
- 报告、图表、运行入口、版本/哈希清单：用于固定本次实际证据和复现。

没有调整训练、S/L、真实碳、DGX功率、随机扰动或服务标准。第一次pilot仅修正汇总层holding_s缺省处理，记录保留。初次core汇总追上独立worker时禁止覆盖保护触发，core_attempt1.log、pipeline_attempt1.log保留；等待120个metrics全部完成后仅恢复汇总，没有重复创建独立样本。最终独立碳核算脚本最初误写gco2字段名，按实际metadata改为gco2e后重新检查，validate_release_attempt1.log与finish_checks_attempt1.log保留；原仿真使用的碳字段及结果未改。报告脚本增加显式UTF-8声明以解决本地Python3.9文件读取报错，reports_release_attempt1.log保留；报告阶段未重算或更改实验数值。测量CPU的H结果可因恢复开销和本机wall变化而改变，未强制匹配旧数值。原正式baseline及此前E/F/G/H均未覆盖。
