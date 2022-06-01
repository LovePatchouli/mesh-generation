# 这个实现的是根据一组外部划分数，遍历所有的内部划分数，并对生成结果保存成png文件（不好的选择）
# 注意生成的方案会非常多（视外部分割数而定），建议更改文件保存位置。

import singularity_pairs as sp
import matplotlib.pyplot as plt
import mesh_evolution
import os


# 外部分割数得的顺序按照论文的顺序来
def pairs_series_generate(ver_list, external_div_list):
    N = external_div_list.copy()
    os.makedirs('{}'.format(N))
    for i in range(1, N[2]):
        n4 = i
        n0 = N[2] - i
        for j in range(1, N[1]):
            n1 = j
            n5 = N[1] - j
            for k in range(1, N[0] - N[2]):
                n2 = k
                n3 = N[0] - N[2] - k

                internal_div_list = [n0, n1, n2, n3, n4, n5]
                sp.plot_mesh(ver_list, internal_div_list)
                plt.savefig("{}/{}.png".format(N,internal_div_list))
                plt.clf()


if __name__ == "__main__":
    a0 = mesh_evolution.node(0, 0)
    a1 = mesh_evolution.node(10, 0)
    a2 = mesh_evolution.node(10, 10)
    a3 = mesh_evolution.node(0, 10)

    v_list = [a0, a1, a2, a3]
    ex_div_list = [12, 6, 7, 11]

    pairs_series_generate(v_list, ex_div_list)