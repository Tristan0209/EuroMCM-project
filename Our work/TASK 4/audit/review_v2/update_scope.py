from pathlib import Path

root = Path(__file__).resolve().parents[2]
specific = {
    'Task4_Implementation_Plan_ZH.md': [
        ('在一个原始1000请求世界、一个种子下，P0/P1新旧事件和control完全一致', '在同一原始世界的1000及7335两种规模、种子2027101800下，P0/P1新旧事件和control完全一致'),
        ('toy截尾和原始1000无截尾等价；完整7335 DES及联合压力尚未验证', 'toy截尾和原始1000/7335单世界无截尾等价；联合压力尚未验证'),
        ('toy与原始1000世界已验证', 'toy与原始1000/7335单世界已验证'),
        ('新观察器对一个原始1000请求世界、一个seed、P0/P1与最终DES逐事件等价；尚未验证全7335轨迹和正式多故障网格', '新观察器对同一原始world在1000/7335规模、一个seed、P0/P1与最终DES逐事件等价；正式多故障/跨月网格尚未验证'),
        ('27项合同（含原始1000世界等价回放）', '27项合同（含原始1000/7335单世界等价回放）'),
    ],
    'Task4_Mathematical_Model_ZH.md': [
        ('窗口/核算toy和原始1000等价', '窗口/核算toy和原始1000/7335单世界等价'),
        ('真实扰动压力、完整7335新DES、多host/PUE正式场景尚未验证', '真实扰动压力、多seed/跨月7335新DES、多host/PUE正式场景尚未验证'),
        ('已实现并验证toy与原始1000无截尾等价', '已实现并验证toy与原始1000/7335单世界无截尾等价'),
        ('已验证一个原始1000请求世界、一个服务seed下两策略的新旧执行事件/control完全一致，共同记录/账本字段及非wall指标atol=1e-8。完整7335 DES与压力网格未验证。全7335路由逐ID已核对，但这不是全7335 DES等价证明。', '已验证同一原始世界在1000及7335规模、seed2027101800下两策略的新旧执行事件/control完全一致，共同记录/账本字段及非wall指标atol=1e-8。全7335路由也已逐ID核对；以上只支持该单世界原始情景，不能推广为24seed、跨月或压力网格已验证。'),
    ],
    'Task4_Execution_Checklist_ZH.md': [
        ('一个原始1000请求世界、一个seed、两策略的新旧事件/control完全一致', '同一原始world在1000/7335规模、seed2027101800、两策略的新旧事件/control完全一致'),
        ('全7335新DES、正式扰动/增长/公平/尾风险/LHS效果', '多seed/压力下7335新DES、正式扰动/增长/公平/尾风险/LHS效果'),
        ('一次原始1000世界的双引擎兼容回放', '同一原始world的1000/7335双引擎兼容回放'),
        ('toy多请求与原始1000，P0/P1', 'toy多请求与原始1000/7335单世界，P0/P1'),
        ('完整7335新DES等价若后续用于更强复用主张，应另做单世界契约核对或从完整保存工件独立重核；当前只验证了1000轨迹。', '当前已完成原始7335单世界等价；若扩展复用主张到其他seed/月份/故障，仍须逐键与完整工件独立重核，不能由单世界推广。'),
    ],
    'Task4_Review_Fixes_ZH.md': [
        ('对一个原始1000请求world、seed2027101800、P0/P1，与最终原引擎分别回放并核算', '对同一原始world在1000和7335规模、seed2027101800、P0/P1，与最终原引擎分别回放并核算'),
        ('不声称全7335 DES已测', '全7335单世界已测，不声称多seed/多故障已测'),
        ('toy长尾、running/queued/inflight/routing/pending、H时完成通过', 'toy长尾、running/queued/inflight/routing/pending、H时完成及原始1000/7335等价通过'),
        ('全7335新DES、生产完整日志压力下的内存/wall', '全7335在其他seed/跨月/压力下的DES、生产完整日志压力下的内存/wall'),
    ],
    'Task4_Readiness_Report.md': [
        ('一个原始1000请求world下两策略双引擎等价', '同一原始world在1000/7335规模下两策略双引擎等价'),
        ('toy+原始1000无截尾等价', 'toy+原始1000/7335单世界无截尾等价'),
        ('完整7335轨迹及所有压力下的新观察器', '多seed/跨月/压力下7335新观察器'),
        ('实现存在，但未做全7335 DES等价或正式长尾/多故障验证；不能把全7335路由验证替代轨迹验证', '原始7335单世界等价已验证；其他seed/跨月及正式长尾/多故障尚未验证，不能从单世界推广'),
        ('一次受控1000世界等价回放', '同一受控原始world的1000/7335等价回放'),
        ('无截尾原始1000轨迹通过并不覆盖所有7335/长尾/多故障路径', '原始1000/7335单世界轨迹通过并不覆盖全部seed/长尾/多故障路径'),
        ('原始1000世界使用seed2027101800，Greedy事件7000、Spatial事件7974', '原始1000和7335规模使用seed2027101800，1000规模Greedy/Spatial事件7000/7974，7335规模51323/54940'),
    ],
    'Task4_Estimated_Runtime.md': [
        ('本轮合同结果约数秒，仅包含全7335推断、toy及1000请求双引擎回放', '本轮27项合同约29s，包含全7335推断、toy及1000/7335单世界双引擎回放'),
    ],
}
for path in root.glob('*.md'):
    content = path.read_text(encoding='utf-8')
    content = content.replace('26项', '27项').replace('26 项', '27 项').replace('attempt_002/', 'attempt_003/')
    content = content.replace("--out 'TASK 4/audit/review_v2/attempt_003'", "--out 'TASK 4/audit/review_v2/attempt_004'")
    for before, after in [('两个合同尝试', '三个合同尝试'), ('两次合同尝试', '三次合同尝试'),
                          ('本轮两个合同attempt', '本轮三个合同attempt'), ('本轮两个attempt', '本轮三个attempt'),
                          ('attempt_001/002均未删除', 'attempt_001/002/003均未删除')]:
        content = content.replace(before, after)
    for before, after in specific.get(path.name, []):
        if before not in content:
            raise ValueError('Expected source phrase absent in ' + path.name + ': ' + before)
        content = content.replace(before, after)
    path.write_text(content, encoding='utf-8')
print('Updated all seven documents to final 27-contract/full-original-world verification scope')
