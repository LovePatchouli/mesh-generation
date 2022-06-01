# ①重新实现了插入双奇异点的分割算法，实现思路相同，但是把分割得到的节点在每一个block中以矩阵的形式存储。
# 这个文件贡献的主要思路也在于此。每个block都可以视为结构化分割，因此block内部用矩阵存储，在每一个block内部就能很轻易地得到节点的拓扑关系。
# block是有限的，因此在block之间的公共边手动把公共节点插入，程序中又更为详细的步骤
# ②实现了laplacian smmothing的经典方法，但是似乎在次外围节点的位置求解中出现了问题。2022/5/25 该问题已经修复，鉴定为手动添加block之间的公共节点的时候有重复，找了整整两个小时bug！ 
# ③实现了基于角度优化的方法，将laplacian smoothing 和angle——based的方法先后使用似乎有不错的效果。

import numpy as np
import mesh_evolution as m_s
import matplotlib.pyplot as plt
import math


class block(object):
    def __init__(self, ver_list, div_list):
        self.vertices = ver_list
        self.div_list = div_list
        self.point_array = []

    # 这个方法是将处理好的顶点放到point_array中
    def add_vertices_to_array(self):
        n0 = self.div_list[0]
        n1 = self.div_list[1]

        self.point_array = [[0 for i in range(n1 + 1)] for j in range(n0 + 1)]
        self.point_array[0][0] = self.vertices[1]
        self.point_array[0][n1] = self.vertices[2]
        self.point_array[n0][n1] = self.vertices[3]
        self.point_array[n0][0] = self.vertices[0]

    def get_node(self, i, j, point):
        self.point_array[i][j] = point

    # 画出array中的点, 可动点为蓝，不可动为红
    def plot_array(self):
        for v in self.point_array:
            for point in v:
                if point.__class__.__name__ != 'int':
                    if point.lock is False:
                        plt.scatter(point.x, point.y, c='b')
                    else:
                        plt.scatter(point.x, point.y, c='r')


class mesh_elem(object):
    def __init__(self):
        self.node_list = []
        self.angle_list = []

    def get_vertices(self, node):
        self.node_list.append(node)

    def plot_elem(self):
        plt.plot([self.node_list[0].x, self.node_list[1].x], [self.node_list[0].y, self.node_list[1].y], c='r')
        plt.plot([self.node_list[1].x, self.node_list[2].x], [self.node_list[1].y, self.node_list[2].y], c='r')
        plt.plot([self.node_list[2].x, self.node_list[3].x], [self.node_list[2].y, self.node_list[3].y], c='r')
        plt.plot([self.node_list[3].x, self.node_list[0].x], [self.node_list[3].y, self.node_list[0].y], c='r')

    # 获取单元内部的四个角度
    def get_internal_angel(self):
        temp = get_angle_from_points(self.node_list[0].x, self.node_list[0].y, self.node_list[3].x, self.node_list[3].y,
                                     self.node_list[1].x, self.node_list[1].y)
        self.angle_list.append(temp)

        temp = get_angle_from_points(self.node_list[1].x, self.node_list[1].y, self.node_list[2].x, self.node_list[2].y,
                                     self.node_list[0].x, self.node_list[0].y)
        self.angle_list.append(temp)

        temp = get_angle_from_points(self.node_list[2].x, self.node_list[2].y, self.node_list[3].x, self.node_list[3].y,
                                     self.node_list[1].x, self.node_list[1].y)
        self.angle_list.append(temp)

        temp = get_angle_from_points(self.node_list[3].x, self.node_list[3].y, self.node_list[0].x, self.node_list[0].y,
                                     self.node_list[2].x, self.node_list[2].y)
        self.angle_list.append(temp)

    # 清除所有已经生成的角度
    def clear_angel(self):
        self.angle_list = []


# 获取两点连线的长度
def get_edge_len_from_two_node_position(x1, y1, x2, y2):
    edge_len = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return edge_len


# 根据三个点获取角度，其中x1，y1为角点
def get_angle_from_points(x1, y1, x2, y2, x3, y3):
    a = get_edge_len_from_two_node_position(x1, y1, x2, y2)
    b = get_edge_len_from_two_node_position(x1, y1, x3, y3)
    c = get_edge_len_from_two_node_position(x2, y2, x3, y3)

    # 用余弦定理求角
    angle = math.degrees(math.acos((a**2 + b**2 - c**2) / (2*a*b)))

    return angle


# 计算∑n（r）*n（r+1）
def sigma(div_list):
    n = div_list
    num = len(n)

    total = 0
    for r in range(num - 1):
        total += 1 / (n[r] * n[r + 1])

    total += 1 / (n[0] * n[num - 1])

    return total


# 解出两个奇异点和奇异点分割点的位置，在x中，分别为b2, xt, xp 中求解其中一个坐标
def solve_formation(ver_list, mid_point_list, div_list):
    a = ver_list
    b = mid_point_list
    n = div_list

    # 下面是方程组
    lhs_1 = [1 / (n[2] * n[0]) + 1 / (n[1] * n[2]), - sigma([n[0], n[1], n[2]]), 0]
    lhs_2 = [(1 / (n[1] * n[3]) + 1 / (n[3] * n[0])), 0, - sigma([n[3], n[1], n[4], n[5], n[0]])]
    lhs_3 = [-sigma([n[2], n[1], n[3], n[0]]), (1 / (n[1] * n[2]) + 1 / (n[0] * n[2])),
             (1 / (n[1] * n[3]) + 1 / (n[3] * n[0]))]

    rhs_1 = - b[1] * (1 / (n[0] * n[1]) + 1 / (n[1] * n[2])) - b[0] * (1 / (n[2] * n[0]) + 1 / (n[0] * n[1])) + a[0] / (
                n[0] * n[1]) + a[5] / (n[1] * n[2]) + a[4] / (n[0] * n[2])
    rhs_2 = - b[3] * (1 / (n[1] * n[3]) + 1 / (n[1] * n[4])) - b[4] * (1 / (n[1] * n[4]) + 1 / (n[4] * n[5])) - b[5] * (
                1 / (n[4] * n[5]) + 1 / (n[5] * n[0])) - b[6] * (1 / (n[0] * n[5]) + 1 / (n[3] * n[0])) + a[5] / (
                        n[1] * n[3]) + a[1] / (n[1] * n[4]) + a[2] / (n[4] * n[5]) + a[3] / (n[0] * n[5]) + a[4] / (
                        n[3] * n[0])
    rhs_3 = b[1] / (n[1] * n[2]) + b[3] / (n[1] * n[3]) + b[6] / (n[3] * n[0]) + b[0] / (n[0] * n[2]) - a[5] * (
                1 / (n[1] * n[2]) + 1 / (n[1] * n[3])) - a[4] * (1 / (n[3] * n[0]) + 1 / (n[0] * n[2]))

    x = np.linalg.solve([lhs_1, lhs_2, lhs_3], [rhs_1, rhs_2, rhs_3])
    return x


def Get_all_nodes(ver_list, div_num):
    a = ver_list.copy()
    n = div_num
    a_4 = m_s.Get_edge_midpoint_position(a[0], a[3], n[1] + n[2], n[3] + n[5])
    a_5 = m_s.Get_edge_midpoint_position(a[0], a[1], n[0] + n[2], n[3] + n[4])

    a.append(a_4)
    a.append(a_5)

    for point in a:
        point.lock = True

    b_1 = m_s.Get_edge_midpoint_position(a[0], a[5], n[0], n[2])
    b_3 = m_s.Get_edge_midpoint_position(a[5], a[1], n[3], n[4])
    b_4 = m_s.Get_edge_midpoint_position(a[1], a[2], n[1], n[5])
    b_5 = m_s.Get_edge_midpoint_position(a[2], a[3], n[4], n[0])
    b_6 = m_s.Get_edge_midpoint_position(a[3], a[4], n[5], n[3])
    b_0 = m_s.Get_edge_midpoint_position(a[0], a[4], n[1], n[2])
    b = [b_0, b_1, m_s.node(0, 0), b_3, b_4, b_5, b_6]

    for point in b:
        point.lock = True

    return [a, b]


def singularity_generate(ver_list, div_num):
    a = ver_list.copy()
    n = div_num
    a_4 = m_s.Get_edge_midpoint_position(a[0], a[3], n[1] + n[2], n[3] + n[5])
    a_5 = m_s.Get_edge_midpoint_position(a[0], a[1], n[0] + n[2], n[3] + n[4])

    a.append(a_4)
    a.append(a_5)

    b_1 = m_s.Get_edge_midpoint_position(a[0], a[5], n[0], n[2])
    b_3 = m_s.Get_edge_midpoint_position(a[5], a[1], n[3], n[4])
    b_4 = m_s.Get_edge_midpoint_position(a[1], a[2], n[1], n[5])
    b_5 = m_s.Get_edge_midpoint_position(a[2], a[3], n[4], n[0])
    b_6 = m_s.Get_edge_midpoint_position(a[3], a[4], n[5], n[3])
    b_0 = m_s.Get_edge_midpoint_position(a[0], a[4], n[1], n[2])
    b = [b_0, b_1, m_s.node(0, 0), b_3, b_4, b_5, b_6]

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

    b2 = m_s.node(singularity_x[0], singularity_y[0])
    xt = m_s.node(singularity_x[1], singularity_y[1])
    xp = m_s.node(singularity_x[2], singularity_y[2])

    return [b2, xt, xp]


def plot_edge_from_next_node(node_list):
    for item in node_list:
        for nod in item.next_node:
            plt.plot([item.x, nod.x], [item.y, nod.y], c='r')
    plt.autoscale(False)
    plt.show()
    plt.cla()


# 简单的拉普拉斯光顺化，nodelist是所有的节点列表， n是迭代次数
def laplacian_smoothing(node_list, n):
    for i in range(n):
        for point in node_list:
            if point.lock is False:
                x = 0
                y = 0
                for nex in point.next_node:
                    x += nex.original_x
                    y += nex.original_y
                x_pos = x / len(point.next_node)
                y_pos = y / len(point.next_node)

                point.x = x_pos
                point.y = y_pos

        # 把光顺后的网格画出来
        # plot_edge_from_next_node(node_list)

        # for item in node_list:
        #     for nod in item.next_node:
        #         plt.plot([item.x, nod.x], [item.y, nod.y], c='r')
        # plt.autoscale(False)
        # plt.show()
        # plt.cla()
        print('第{}次迭代完毕'.format(i+1))

        for item in node_list:
            item.original_x = item.x
            item.original_y = item.y


# 角平分线调整法, nod是待调整的节点的位置，elem是该节点的一个邻接单元, 这里实现的仅仅是一个单元中的调整
def get_new_position(nod, elem):
    nod_lis = elem.node_list
    index = nod_lis.index(nod)

    # 根据取余的方法得到另外的几个点，因为在elem中，node是有序的
    next_node = nod_lis[(index+1) % 4]
    opponent_node = nod_lis[(index+2) % 4]
    last_node = nod_lis[(index+3) % 4]

    alph1 = get_angle_from_points(opponent_node.x, opponent_node.y, next_node.x, next_node.y, nod.x, nod.y)
    alph2 = get_angle_from_points(opponent_node.x, opponent_node.y, last_node.x, last_node.y, nod.x, nod.y)

    beta = (alph2 - alph1) / 2 / 180 * math.pi

    new_x = opponent_node.x + (nod.x - opponent_node.x)*math.cos(beta) - (nod.y - opponent_node.y)*math.sin(beta)
    new_y = opponent_node.y + (nod.x - opponent_node.x)*math.sin(beta) + (nod.y - opponent_node.y)*math.cos(beta)

    return new_x, new_y


# 质量评估
def get_quality_of_mesh_based_on_angel(elem_list):
    max_angel = -1
    min_angle = 100
    # 下面两个变量分别揭露大于120和小于60的角的个数
    n1 = 0
    n2 = 0
    for item in elem_list:
        item.get_internal_angel()
        if min(item.angle_list) < 60:
            n1 += 1
        if max(item.angle_list) > 120:
            n2 += 1
        min_angle = min(min_angle, min(item.angle_list))
        max_angel = max(max_angel, max(item.angle_list))
        item.clear_angel()
    print(max_angel, min_angle)
    print("小于60度的单元有{}， 大于120的单元有{}".format(n1, n2))

    return 0


def get_new_position_from_around_nodes(mid_node):  # 角度优化，around nodes指的是调整节点周围所有单元的节点
    ranked_next_node_list = []

    unranked_elem_list = mid_node.around_elem.copy()
    start_elem = unranked_elem_list.pop()

    mid_node_index = start_elem.node_list.index(mid_node)

    temp_elem = start_elem
    cur_index = (mid_node_index + 1) % 4

    while True:
        next_index = (cur_index + 1) % 4
        ranked_next_node_list.append(temp_elem.node_list[cur_index])
        ranked_next_node_list.append(temp_elem.node_list[next_index])

        cur_index = (next_index + 1) % 4

        if temp_elem.node_list[cur_index] in ranked_next_node_list:  # 如果下个点已经在有序节点列表之内的话，那么说明已经添加完一周了，可以结束了
            break
        else:
            for elem in unranked_elem_list:

                # 在未排序单元中，找到下一个单元
                if temp_elem.node_list[cur_index] in elem.node_list:
                    cur_index = elem.node_list.index(temp_elem.node_list[cur_index])
                    temp_elem = elem
                    unranked_elem_list.remove(elem)
                    break

    # 一段测试
    # for temp in ranked_next_node_list:
    #     plt.scatter(temp.x, temp.y, c='y')
    #
    # plt.scatter(mid_node.x, mid_node.y, c='r')
    # plt.show()
    # print(len(ranked_next_node_list))
    # 开始计算调整点的位置

    modified_node_position = []  # 这里是调整好的点的位置
    node_num = len(ranked_next_node_list)   # node num 指的是周围节点的个数
    for i in range(node_num):
        front_node = ranked_next_node_list[i]
        cur_node = ranked_next_node_list[(i+1) % node_num]
        next_node = ranked_next_node_list[(i+2) % node_num]  # 这里是连续的三个外围点

        '''现在有了三个外围点和中点，可以开始计算呢调整点的位置'''
        alph1 = get_angle_from_points(cur_node.x, cur_node.y, front_node.x, front_node.y, mid_node.x, mid_node.y)
        alph2 = get_angle_from_points(cur_node.x, cur_node.y, next_node.x, next_node.y, mid_node.x, mid_node.y)

        beta = (alph2 - alph1) / 2 / 180 * math.pi

        new_x = cur_node.x + (mid_node.x - cur_node.x) * math.cos(beta) - (mid_node.y - cur_node.y) * math.sin(
            beta)
        new_y = cur_node.y + (mid_node.x - cur_node.x) * math.sin(beta) + (mid_node.y - cur_node.y) * math.cos(
            beta)

        modified_node_position.append((new_x, new_y))

    x_sum = 0
    y_sum = 0

    for pos in modified_node_position:
        x_sum += pos[0]
        y_sum += pos[1]

    new_x = x_sum / node_num
    new_y = y_sum / node_num

    mid_node.x = new_x
    mid_node.y = new_y

    return 0


def main():
    # 初始给出的四个顶点和分割数
    v_list = []
    d_list = []

    # a0 = m_s.node(0, 0)
    # a1 = m_s.node(10, 0)
    # a2 = m_s.node(10, 10)
    # a3 = m_s.node(0, 10)
    #
    # v_list = [a0, a1, a2, a3]
    # d_list = [3, 4, 3, 4, 4, 4]
    #
    d_list = [6, 1, 3, 4, 1, 5]

    # d_list = [4, 2, 2, 3, 3, 4]

    a0 = m_s.node(0, 0)
    a1 = m_s.node(10, -2)
    a2 = m_s.node(15, 5)
    a3 = m_s.node(10, 17)

    v_list = [a0, a1, a2, a3]
    # d_list = [6, 1, 3, 4, 1, 5]

    n = d_list
    '''
    ----------数据与程序的分割线----------
    '''
    # 这里的temp 是所有外部分割点的点列，理所应当的，lock值为true
    temp = Get_all_nodes(v_list, d_list)
    a = temp[0]
    b = temp[1]

    # 这里的temp 是内部奇异点和奇异点的分割点b2, xt, xp,并且lock值为false
    temp = singularity_generate(v_list, d_list)
    b[2] = temp[0]
    xt = temp[1]
    xp = temp[2]

    block_1 = block([b[6], a[3], b[5], xp], [n[5], n[0], n[5], n[0]])
    block_2 = block([xp, b[5], a[2], b[4]], [n[5], n[4], n[5], n[4]])
    block_3 = block([b[3], xp, b[4], a[1]], [n[1], n[4], n[1], n[4]])
    block_4 = block([a[5], b[2], xp, b[3]], [n[1], n[3], n[1], n[3]])
    block_5 = block([b[1], xt, b[2], a[5]], [n[1], n[2], n[1], n[2]])
    block_6 = block([a[0], b[0], xt, b[1]], [n[1], n[0], n[1], n[0]])
    block_7 = block([b[0], a[4], b[2], xt], [n[2], n[0], n[2], n[0]])
    block_8 = block([a[4], b[6], xp, b[2]], [n[3], n[0], n[3], n[0]])

    blocks = [block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8]

    # 测试block中的顶点是否符合要求
    # for point in block_1.vertices:
    #     plt.scatter(point.x, point.y)
    # plt.show()

    for blo in blocks:
        blo.add_vertices_to_array()
    print('block的顶点添加完毕')

    '''手动向block之间添加公共边的节点'''
    # b5 xp
    for i in range(n[5] - 1):
        x_p = b[5].x + (xp.x - b[5].x) / n[5] * (i + 1)
        y_p = b[5].y + (xp.y - b[5].y) / n[5] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_1.get_node(i + 1, n[0], temp)
        block_2.get_node(i + 1, 0, temp)

    # xp b4
    for i in range(n[4] - 1):
        x_p = xp.x + (b[4].x - xp.x) / n[4] * (i + 1)
        y_p = xp.y + (b[4].y - xp.y) / n[4] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_2.get_node(n[5], i + 1, temp)
        block_3.get_node(0, i + 1, temp)

    # xp, b3
    for i in range(n[1] - 1):
        x_p = xp.x + (b[3].x - xp.x) / n[1] * (i + 1)
        y_p = xp.y + (b[3].y - xp.y) / n[1] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_3.get_node(i + 1, 0, temp)
        block_4.get_node(i + 1, n[3], temp)

    # b2 a5
    for i in range(n[1] - 1):
        x_p = b[2].x + (a[5].x - b[2].x) / n[1] * (i + 1)
        y_p = b[2].y + (a[5].y - b[2].y) / n[1] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_4.get_node(i + 1, 0, temp)
        block_5.get_node(i + 1, n[2], temp)

    # xt b1
    for i in range(n[1] - 1):
        x_p = xt.x + (b[1].x - xt.x) / n[1] * (i + 1)
        y_p = xt.y + (b[1].y - xt.y) / n[1] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_6.get_node(i + 1, n[0], temp)
        block_5.get_node(i + 1, 0, temp)

    # xt b0
    for i in range(n[0] - 1):
        x_p = b[0].x + (xt.x - b[0].x) / n[0] * (i + 1)
        y_p = b[0].y + (xt.y - b[0].y) / n[0] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_7.get_node(n[2], i + 1, temp)
        block_6.get_node(0, i + 1, temp)

    # a[4] b[2]
    for i in range(n[0] - 1):
        x_p = a[4].x + (b[2].x - a[4].x) / n[0] * (i + 1)
        y_p = a[4].y + (b[2].y - a[4].y) / n[0] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_7.get_node(0, i + 1, temp)
        block_8.get_node(n[3], i + 1, temp)

    # b[6], xp
    for i in range(n[0] - 1):
        x_p = b[6].x + (xp.x - b[6].x) / n[0] * (i + 1)
        y_p = b[6].y + (xp.y - b[6].y) / n[0] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_8.get_node(0, i + 1, temp)
        block_1.get_node(n[5], i + 1, temp)

    # xt b[2]
    for i in range(n[2] - 1):
        x_p = xt.x + (b[2].x - xt.x) / n[2] * (i + 1)
        y_p = xt.y + (b[2].y - xt.y) / n[2] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_7.get_node(n[2] - i - 1, n[0], temp)
        block_5.get_node(0, i + 1, temp)

    # b2 xp
    for i in range(n[3] - 1):
        x_p = b[2].x + (xp.x - b[2].x) / n[3] * (i + 1)
        y_p = b[2].y + (xp.y - b[2].y) / n[3] * (i + 1)
        temp = m_s.node(x_p, y_p)
        block_8.get_node(n[3] - i - 1, n[0], temp)
        block_4.get_node(0, i + 1, temp)

    '''添加外围的固定点'''
    # a3 b[5]
    for i in range(n[0]-1):
        x_p = a[3].x + (b[5].x - a[3].x) / n[0] * (i + 1)
        y_p = a[3].y + (b[5].y - a[3].y) / n[0] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_1.get_node(0, i + 1, temp)

    # b[5] a[2]
    for i in range(n[4]-1):
        x_p = b[5].x + (a[2].x - b[5].x) / n[4] * (i + 1)
        y_p = b[5].y + (a[2].y - b[5].y) / n[4] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_2.get_node(0, i + 1, temp)

    # a[0], b[1]
    for i in range(n[0]-1):
        x_p = a[0].x + (b[1].x - a[0].x) / n[0] * (i + 1)
        y_p = a[0].y + (b[1].y - a[0].y) / n[0] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_6.get_node(n[1], i + 1, temp)

    # b[1] a[5]
    for i in range(n[2]-1):
        x_p = b[1].x + (a[5].x - b[1].x) / n[2] * (i + 1)
        y_p = b[1].y + (a[5].y - b[1].y) / n[2] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_5.get_node(n[1], i + 1, temp)

    # a[5] b[3]
    for i in range(n[3]-1):
        x_p = a[5].x + (b[3].x - a[5].x) / n[3] * (i + 1)
        y_p = a[5].y + (b[3].y - a[5].y) / n[3] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_4.get_node(n[1], i + 1, temp)

    # b[3] a[1]
    for i in range(n[4]-1):
        x_p = b[3].x + (a[1].x - b[3].x) / n[4] * (i + 1)
        y_p = b[3].y + (a[1].y - b[3].y) / n[4] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_3.get_node(n[1], i + 1, temp)

    # a[3] b[6]
    for i in range(n[5]-1):
        x_p = a[3].x + (b[6].x - a[3].x) / n[5] * (i + 1)
        y_p = a[3].y + (b[6].y - a[3].y) / n[5] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_1.get_node(i + 1, 0, temp)

    # b[6] a[4]
    for i in range(n[3]-1):
        x_p = b[6].x + (a[4].x - b[6].x) / n[3] * (i + 1)
        y_p = b[6].y + (a[4].y - b[6].y) / n[3] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_8.get_node(i + 1, 0, temp)

    # a[4] b[0]
    for i in range(n[2]-1):
        x_p = a[4].x + (b[0].x - a[4].x) / n[2] * (i + 1)
        y_p = a[4].y + (b[0].y - a[4].y) / n[2] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_7.get_node(i + 1, 0, temp)

    # b[0] a[0]
    for i in range(n[1]-1):
        x_p = b[0].x + (a[0].x - b[0].x) / n[1] * (i + 1)
        y_p = b[0].y + (a[0].y - b[0].y) / n[1] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_6.get_node(i + 1, 0, temp)

    # a[2] b[4]
    for i in range(n[5]-1):
        x_p = a[2].x + (b[4].x - a[2].x) / n[5] * (i + 1)
        y_p = a[2].y + (b[4].y - a[2].y) / n[5] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_2.get_node(i + 1, n[4], temp)

    # b[4] a[1]
    for i in range(n[1]-1):
        x_p = b[4].x + (a[1].x - b[4].x) / n[1] * (i + 1)
        y_p = b[4].y + (a[1].y - b[4].y) / n[1] * (i + 1)
        temp = m_s.node(x_p, y_p)
        temp.lock = True
        block_3.get_node(i + 1, n[4], temp)

    print('blocks的外围节点全部创建完毕')

    # test 5.21 检查处bug并且修改成功
    # for item in blocks:
    #     array = item.point_array
    #     print(array)
    #     for i in array:
    #         for point in i:
    #             if type(point) != int:
    #                 plt.scatter(point.x, point.y, c='r')

    # plt.show()
    '''------------------外围的节点全部创建晚比，下面是要对每个block创建内部的节点-----------'''

    for item in blocks:
        array = item.point_array
        for i in range(1, item.div_list[0]):
            for j in range(1, item.div_list[1]):
                n1 = (array[i][0].x, array[i][0].y)
                n2 = (array[i][item.div_list[1]].x, array[i][item.div_list[1]].y)

                m1 = (array[0][j].x, array[0][j].y)
                m2 = (array[item.div_list[0]][j].x, array[item.div_list[0]][j].y)

                lhs_1 = [(n2[1]-n1[1]) / (n2[0]-n1[0]), -1]
                lhs_2 = [(m2[1]-m1[1]) / (m2[0]-m1[0]), -1]

                rhs_1 = (n2[1]-n1[1]) / (n2[0]-n1[0]) * n1[0] - n1[1]
                rhs_2 = (m2[1]-m1[1]) / (m2[0]-m1[0]) * m1[0] - m1[1]

                p = np.linalg.solve([lhs_1, lhs_2], [rhs_1, rhs_2])
                item.get_node(i, j, m_s.node(p[0], p[1]))

    '''内部的节点添加完成了呢， 下面就是每个点添加他们的邻接点'''

    # 先建立一个点集罢,处理好邻接点之后加入该点集
    node_list = []

    for item in blocks:
        array = item.point_array
        for i in range(item.div_list[0]+1):
            for j in range(item.div_list[1]+1):
                if i+1 <= item.div_list[0]:
                    array[i][j].get_next_node(array[i+1][j])
                if i-1 > -1:
                    array[i][j].get_next_node(array[i-1][j])
                if j-1 > -1:
                    array[i][j].get_next_node(array[i][j-1])
                if j+1 <= item.div_list[1]:
                    array[i][j].get_next_node(array[i][j+1])

                if array[i][j] not in node_list:
                    node_list.append(array[i][j])

    print("邻接点生成完毕")

    # 再建立一个网格单元集罢，生成每个网格单元并且加入，单元的节点都是有序的，从左上面的开始顺时针转
    elem_list = []
    for item in blocks:
        array = item.point_array
        for i in range(item.div_list[0]):
            for j in range(item.div_list[1]):
                elem = mesh_elem()
                elem.get_vertices(array[i][j])
                elem.get_vertices(array[i][j+1])
                elem.get_vertices(array[i+1][j+1])
                elem.get_vertices(array[i+1][j])

                elem_list.append(elem)

    '''原始的分割'''
    print('原始分割生成完毕')

    # 建立某一个点的邻接单元
    for item in elem_list:
        for nod in item.node_list:
            nod.around_elem.append(item)

    print("节点的邻接单元添加完毕")

    '''---------------------以上为网格拓扑结构的建立----------------------------'''

    for elem in elem_list:
        elem.plot_elem()
    plt.autoscale(False)
    plt.gca().set_aspect(1)
    plt.show()
    plt.cla()
    print('初始分割单元绘制完毕')

    # 初始分割网格的质量
    get_quality_of_mesh_based_on_angel(elem_list)

    '''--------------------------------------以下是基于基于角度光顺化,并且查看结果------------------------------------------------'''
    # laplacian_smoothing(node_list, 9)
    # for i in range(1):
    #     for nod in node_list:
    #         if nod.lock is True:
    #             continue
    #
    #         num = nod.get_around_elem_num()
    #         temp_x = 0
    #         temp_y = 0
    #         # 首先遍历某个节点的所有的邻接单元，顺序并不重要
    #         for elem in nod.around_elem:
    #             temp = get_new_position(nod, elem)
    #             temp_x += temp[0]
    #             temp_y += temp[1]
    #
    #         new_x = temp_x / num
    #         new_y = temp_y / num
    #
    #         nod.x = new_x
    #         nod.y = new_y
    #
    # for elem in elem_list:
    #     elem.plot_elem()
    # plt.autoscale(False)
    # plt.gca().set_aspect(1)
    # plt.show()
    # plt.cla()
    # print('角度优化绘制完毕')
    #
    # get_quality_of_mesh_based_on_angel(elem_list)

    '''--------------------------以下统计最小的角度，最大的角度，查看laplacian光顺的结果-----------------------------------'''
    # max_angel = -1
    # min_angle = 100
    # # 下面两个变量分别揭露大于120和小于60的角的个数
    # n1 = 0
    # n2 = 0
    # for item in elem_list:
    #     item.get_internal_angel()
    #     if min(item.angle_list) < 60:
    #         n1 += 1
    #     if max(item.angle_list) > 120:
    #         n2 += 1
    #     min_angle = min(min_angle, min(item.angle_list))
    #     max_angel = max(max_angel, max(item.angle_list))
    #     item.clear_angel()
    # print(max_angel, min_angle)
    # print("优化前小于60度的单元有{}， 大于120的单元有{}".format(n1, n2))




    # 看看laplacian smoothing后的结果回事怎样
    # laplacian_smoothing(node_list, 5)
    # max_angel = -1
    # min_angle = 100
    # n1 = 0
    # n2 = 0
    # for item in elem_list:
    #     item.get_internal_angel()
    #     if min(item.angle_list) < 60:
    #         n1 += 1
    #     if max(item.angle_list) > 120:
    #         n2 += 1
    #     min_angle = min(min_angle, min(item.angle_list))
    #     max_angel = max(max_angel, max(item.angle_list))
    #     item.clear_angel()
    #
    # print(max_angel, min_angle)
    # print("优化后小于60度的单元有{}， 大于120的单元有{}".format(n1, n2))
    #
    # # '''这是对每个单元画图，验证单元是否全部遍历'''
    # for elem in elem_list:
    #     elem.plot_elem()
    # plt.autoscale(False)
    # plt.gca().set_aspect(1)
    # plt.show()
    # plt.cla()
    # print('光顺化后单元绘制完毕')

    '''根据最大最小角研究完毕----------------------------------------------------------------'''


    '''这是连接邻接点画图'''
    # plot_edge_from_next_node(node_list)
    # for item in node_list:
    #     for nod in item.next_node:
    #         plt.plot([item.x, nod.x], [item.y, nod.y], c='r')
    # plt.autoscale(False)
    # plt.show()
    # plt.cla()

    # '''拉普拉斯光顺化'''
    # laplacian_smoothing(node_list, 9)

    # 这一段连线重构后写在函数里简化了
    # for item in node_list:
    #     for i in item.next_node:
    #         plt.plot([item.x, i.x], [item.y, i.y])

    # 取特殊点检测邻接点所求是否正确
    # for item in b[2].next_node:
    #     plt.scatter(item.x, item.y, s=50, c='y')
    # print(b[2].next_node)

    # 测试公共边的节点是否是同一个
    # (block_1.point_array[2][n[0]]).x = 100
    # print(type(block_1.point_array[2][n[0]]))
    # print(block_1.point_array[2][n[0]].x, block_2.point_array[2][0].x)

    # for i in range(n[])

    # 画出所有的节点，发现符合的
    # for blo in blocks:
    #     blo.plot_array()


    '''---------------正确的角度优化----------------------'''

    # get_new_position_from_around_nodes(xp)
    laplacian_smoothing(node_list, 5)  # 先来一个拉式光顺

    # for i in range(1):  # 遍历所有节点进行角度优化
    #     for nod in node_list:
    #         if nod.lock is False:
    #             get_new_position_from_around_nodes(nod)

    for elem in elem_list:
        elem.plot_elem()
    plt.autoscale(False)
    plt.gca().set_aspect(1)
    plt.show()
    plt.cla()
    get_quality_of_mesh_based_on_angel(elem_list)


if __name__ == '__main__':
    main()
