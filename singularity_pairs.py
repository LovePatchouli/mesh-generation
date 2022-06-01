# 实现了Li的‘Multiblock mesh refinement by adding mesh singularities‘中的内容。
# 主要是对一个四边形区域进行添加奇异点对进行划分。求解奇异点位置，径向边上节点的位置，绘制划分结果


import numpy as np
import mesh_evolution
import matplotlib.pyplot as plt


# 创建一个和block， 里面含有四个顶点和四个边的分割数，顺时针依此定义，这里的block代表的是双奇异点分割得到的分块
class block(object):
    def __init__(self, ver_list, div_list):
        self.vertices = ver_list
        self.div_list = div_list
        self.point_array = []


# 计算∑n（r）*n（r+1）
def sigma(div_list):
    n = div_list
    num = len(n)

    total = 0
    for r in range(num-1):
        total += 1 / (n[r] * n[r+1])

    total += 1/(n[0] * n[num-1])

    return total


# 解出两个奇异点和奇异点分割点的位置，在x中，分别为b2, xt, xp 中求解其中一个坐标
def solve_formation(ver_list, mid_point_list, div_list):
    a = ver_list
    b = mid_point_list
    n = div_list

    # 下面是方程组
    lhs_1 = [1 / (n[2] * n[0]) + 1 / (n[1] * n[2]), - sigma([n[0], n[1], n[2]]), 0]
    lhs_2 = [(1 / (n[1] * n[3]) + 1 / (n[3] * n[0])), 0, - sigma([n[3], n[1], n[4], n[5], n[0]])]
    lhs_3 = [-sigma([n[2], n[1], n[3], n[0]]), (1 / (n[1] * n[2]) + 1 / (n[0] * n[2])), (1 / (n[1] * n[3]) + 1 / (n[3] * n[0]))]

    rhs_1 = - b[1] * (1 / (n[0]*n[1]) + 1/(n[1]*n[2])) - b[0] * (1 / (n[2]*n[0]) + 1/(n[0]*n[1])) + a[0] / (n[0] * n[1]) + a[5] / (n[1] * n[2]) + a[4] / (n[0] * n[2])
    rhs_2 = - b[3] * (1 / (n[1]*n[3]) + 1/(n[1]*n[4])) - b[4] * (1 / (n[1]*n[4]) + 1/(n[4]*n[5])) - b[5] * (1 / (n[4]*n[5]) + 1/(n[5]*n[0])) - b[6] * (1 / (n[0]*n[5]) + 1/(n[3]*n[0])) + a[5] / (n[1]*n[3]) + a[1] / (n[1]*n[4]) + a[2] / (n[4]*n[5]) + a[3] / (n[0]*n[5]) + a[4] / (n[3]*n[0])
    rhs_3 = b[1] / (n[1]*n[2]) + b[3] / (n[1]*n[3]) + b[6] / (n[3]*n[0]) + b[0] / (n[0]*n[2]) - a[5] * (1 / (n[1]*n[2]) + 1/(n[1]*n[3])) - a[4] * (1 / (n[3]*n[0]) + 1/(n[0]*n[2]))

    x = np.linalg.solve([lhs_1, lhs_2, lhs_3], [rhs_1, rhs_2, rhs_3])
    return x


# 这里对上面的点坐标计算进行测试
# def test_1():
#     a = [0,10,10,0,0,5]
#     b = [0,2,0,8,10,6,0]
#     n = [1,1,1,1,3,3]
#
#     x = solve_formation(a,b,n)
#     print(x)


# 这里得到的是初始的边界上的点
def Get_all_nodes(ver_list, div_num):
    a = ver_list.copy()
    n = div_num
    a_4 = mesh_evolution.Get_edge_midpoint_position(a[0], a[3], n[1] + n[2], n[3] + n[5])
    a_5 = mesh_evolution.Get_edge_midpoint_position(a[0], a[1], n[0] + n[2], n[3] + n[4])

    a.append(a_4)
    a.append(a_5)

    for point in a:
        point.lock = True

    b_1 = mesh_evolution.Get_edge_midpoint_position(a[0], a[5], n[0], n[2])
    b_3 = mesh_evolution.Get_edge_midpoint_position(a[5], a[1], n[3], n[4])
    b_4 = mesh_evolution.Get_edge_midpoint_position(a[1], a[2], n[1], n[5])
    b_5 = mesh_evolution.Get_edge_midpoint_position(a[2], a[3], n[4], n[0])
    b_6 = mesh_evolution.Get_edge_midpoint_position(a[3], a[4], n[5], n[3])
    b_0 = mesh_evolution.Get_edge_midpoint_position(a[0], a[4], n[1], n[2])
    b = [b_0, b_1, mesh_evolution.node(0, 0), b_3, b_4, b_5, b_6]

    return [a, b]


# 这里是生成奇异点位置， 其中所有节点的定义都是和mesh——evolution中保持一致，verlist中为【a1， a2.。。】
def singularity_generate(ver_list, div_num):
    a = ver_list.copy()
    n = div_num
    a_4 = mesh_evolution.Get_edge_midpoint_position(a[0], a[3], n[1]+n[2], n[3] + n[5])
    a_5 = mesh_evolution.Get_edge_midpoint_position(a[0], a[1], n[0] + n[2], n[3] + n[4])

    a.append(a_4)
    a.append(a_5)

    b_1 = mesh_evolution.Get_edge_midpoint_position(a[0], a[5], n[0], n[2])
    b_3 = mesh_evolution.Get_edge_midpoint_position(a[5], a[1], n[3], n[4])
    b_4 = mesh_evolution.Get_edge_midpoint_position(a[1], a[2], n[1], n[5])
    b_5 = mesh_evolution.Get_edge_midpoint_position(a[2], a[3], n[4], n[0])
    b_6 = mesh_evolution.Get_edge_midpoint_position(a[3], a[4], n[5], n[3])
    b_0 = mesh_evolution.Get_edge_midpoint_position(a[0], a[4], n[1], n[2])
    b = [b_0, b_1, mesh_evolution.node(0, 0), b_3, b_4, b_5, b_6]

    a_x = []
    a_y = []
    b_x = []
    b_y = []
    for point in a:
        a_x.append(point.x)
        a_y.append(point.y)

    for point in b:
        b_x.append(point.x)
        b_y.append(point.y)

    singularity_x = solve_formation(a_x, b_x, div_num)
    singularity_y = solve_formation(a_y, b_y, div_num)

    b2 = mesh_evolution.node(singularity_x[0], singularity_y[0])
    xt = mesh_evolution.node(singularity_x[1], singularity_y[1])
    xp = mesh_evolution.node(singularity_x[2], singularity_y[2])

    return [b2, xt, xp]


def plot_block(block_item):
    ver_list = block_item.vertices
    temp_list = block_item.div_list
    mesh_evolution.plot_outline(ver_list)
    if temp_list[0] == 1 and temp_list[1] == 1:
        return

    elif temp_list[0] == 1:
        for i in range(1, temp_list[1]):
            up_side_point = mesh_evolution.Get_edge_midpoint_position(ver_list[1], ver_list[2], i, temp_list[1]-i)
            bot_side_point = mesh_evolution.Get_edge_midpoint_position(ver_list[0], ver_list[3], i, temp_list[1]-i)
            plt.plot([up_side_point.x, bot_side_point.x], [up_side_point.y, bot_side_point.y], c='r')

    elif temp_list[1] == 1:
        for i in range(1, temp_list[0]):
            up_side_point = mesh_evolution.Get_edge_midpoint_position(ver_list[0], ver_list[1], i, temp_list[0]-i)
            bot_side_point = mesh_evolution.Get_edge_midpoint_position(ver_list[3], ver_list[2], i, temp_list[0]-i)
            plt.plot([up_side_point.x, bot_side_point.x], [up_side_point.y, bot_side_point.y], c='r')

    else:
        div_list = [(1, temp_list[0]-1), (1, temp_list[1]-1), (temp_list[0]-1, 1), (temp_list[1]-1, 1)]
        mesh_evolution.plot_mesh(ver_list, div_list)


def plot_mesh(ver_list, div_num):
    n = div_num

    all_nodes = Get_all_nodes(ver_list, div_num)
    a = all_nodes[0]
    b = all_nodes[1]

    added_nodes = singularity_generate(ver_list, div_num)
    b[2] = added_nodes[0]
    xt = added_nodes[1]
    xp = added_nodes[2]

    block_1 = block([a[3], b[5], xp, b[6]], [n[0], n[5], n[0], n[5]])
    block_2 = block([b[5], a[2], b[4], xp], [n[4], n[5], n[4], n[5]])
    block_3 = block([xp, b[4], a[1], b[3]], [n[4], n[1], n[4], n[1]])
    block_4 = block([b[2], xp, b[3], a[5]], [n[3], n[1], n[3], n[1]])
    block_5 = block([xt, b[2], a[5], b[1]], [n[2], n[1], n[2], n[1]])
    block_6 = block([a[0], b[0], xt, b[1]], [n[1], n[0], n[1], n[0]])
    block_7 = block([b[0], a[4], b[2], xt], [n[2], n[0], n[2], n[0]])
    block_8 = block([a[4], b[6], xp, b[2]], [n[3], n[0], n[3], n[0]])

    blocks = [block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8]

    for items in blocks:
        plot_block(items)


def test_2():
    a0 = mesh_evolution.node(0, 0)
    a1 = mesh_evolution.node(10, 0)
    a2 = mesh_evolution.node(10, 10)
    a3 = mesh_evolution.node(0, 10)
    a4 = mesh_evolution.node(0, 5)
    a5 = mesh_evolution.node(5, 0)

    ver_list = [a0, a1, a2, a3]
    # div_list = [1, 1, 1, 1, 1, 1]
    # div_list = [3, 3, 3, 2, 4, 3]
    div_list = [6, 1, 3, 4, 1, 5]
    div_list = [4, 4, 1, 4, 3, 2]
    div_list = [3, 4, 3, 4, 4, 4]


    plot_mesh(ver_list, div_list)


def main():
    a0 = mesh_evolution.node(0, 0)
    a1 = mesh_evolution.node(10, 0)
    a2 = mesh_evolution.node(10, 10)
    a3 = mesh_evolution.node(0, 10)

    ver_list = [a0, a1, a2, a3]
    div_list = [6, 1, 3, 4, 1, 5]



if __name__ == '__main__':
    test_2()
    plt.gca().set_aspect(1)
    plt.show()



