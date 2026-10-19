from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMPLE = HERE.parent
BS = chr(92)
BLOCKS = [
    r'''Q_{router}=@frac{1}{N}@sum_i q_{i,m_i},@qquad
P_{hist}=@frac{1}{N}@sum_i@mathbf{1}(q_{i,m_i}@ge 0.5),@qquad
P_{valid}=@frac{1}{N}@sum_i I_i@mathbf{1}(q_{i,m_i}@ge 0.5),@qquad
V=@frac{1}{N}@sum_i(1-I_i).''',
    r'''@Delta Q=@sum_g(w_g^D-w_g^B)@mu_g^B+@sum_g w_g^D(@mu_g^D-@mu_g^B).''',
    r'''@Delta Q_b=@frac{|F_b|}{7335}@overline{@delta}_{F_b}.''',
    r'''C_S^{lin}(N)=C_{idle}+Nc_0,@qquad c_0=@frac{C_S(N_0)-C_{idle}}{N_0},@qquad
r_{lin}=@frac{C_G(N_0)-C_S(N_0)}{C_S(N_0)-C_{idle}}.''',
    r'''@Delta C_{active}=(E_1-E_0)@bar I_0+E_1(@bar I_1-@bar I_0).''',
    r'''Q_{router,g}=@frac{1}{N_g}@sum_{i@in g}q_{i,m_i},@qquad
Q_{AllL,g}=@frac{1}{N_g}@sum_{i@in g}q_{i,L},@qquad
@Delta Q_g=@frac{1}{N_g}@sum_{i@in g}(q_{i,m_i}-q_{i,L}).''',
    r'''c_0=@frac{C_S(N_0)-C_{idle}}{N_0},@qquad
r_{lin}=@frac{C_G(N_0)-C_S(N_0)}{C_S(N_0)-C_{idle}}.'''
]


def replace_blocks(path, blocks, template=False):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    out, count, pos = [], 0, 0
    while pos < len(lines):
        if lines[pos] == '[':
            pos += 1
            while lines[pos] != ']':
                pos += 1
            value = blocks[count].replace('@', BS)
            if template:
                value = value.replace('{', '{{').replace('}', '}}')
            out.extend(['$$', value, '$$'])
            count += 1
        else:
            out.append(lines[pos])
        pos += 1
    assert count == len(blocks), (path, count)
    path.write_text('\n'.join(out)+'\n', encoding='utf-8')


if __name__ == '__main__':
    replace_blocks(SIMPLE/'Task4_Simple_Report_Final_ZH.md', BLOCKS[:6])
    replace_blocks(SIMPLE/'Task4_Paper_Outline_Final_EN.md', BLOCKS[6:])
    replace_blocks(HERE/'write_final_reports.py', BLOCKS, template=True)
    print('PASS: six report equations and one outline equation; template repaired')
