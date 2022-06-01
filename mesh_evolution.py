# 实现Li的‘Quad mesh generation for k-sided faces and hex mesh
# generation for trivalent 1 polyhedra ’中的内容，能够实现对一个k—sided多边形区域的网格划分


import numpy as np
import matplotlib.pyplot as plt


class node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        # 以列表的形式存储邻接点
        self.next_node = []
        # lock代表是否是固定点，这里的False代表默认是不固定的
        self.lock = False
        self.access = False
        self.around_elem = []
        self.around_elem_node = [] # 这个around node 事角度优化里围绕在待调整节点外围的一圈点，包括邻接点和对角点，通过外部函数get_around_elem_node得到，这个函数在store——node——as——arrary里

    def get_next_node(self, n):
        if n not in self.next_node:
            self.next_node.append(n)

    def get_around_elem_num(self):
        return len(self.around_elem)





def Get_external_division_number(number_list):
    division_number = number_list
    return division_number


# 获取边上的中点（非几何）如图:  start___u______mid____v_____end
def Get_edge_midpoint_position(start, end, u, v):
    x = start.x + (-start.x + end.x) * u / (u + v)
    y = start.y + (-start.y + end.y) * u / (u + v)

    return node(x, y)


# 根据顶点列表以及划分数列表获取边的中点
# 其中div 的格式为[(), (),...]
def Get_mid_point_from_vertex(ver_list, div_list):
    num = len(ver_list)
    mid_point = []

    for i in range(0, num-1):
        edge_mid_point = Get_edge_midpoint_position(ver_list[i], ver_list[i+1], div_list[i][0], div_list[i][1])
        mid_point.append(edge_mid_point)

    last = Get_edge_midpoint_position(ver_list[num-1], ver_list[0], div_list[num-1][0], div_list[num-1][1])
    mid_point.append(last)

    return mid_point


# vertices_list = [vertex0,...]
# edge_points_list = [midpoint0,...]
# division_number_list = [(),...]
def Get_face_midpoint_position(vertices_list, edge_points_list, division_number_list):
    up = 0
    bottom = 0
    edge_num = len(vertices_list)

    # 分别计算分子分母 顺时针方向
    for i in range(1, edge_num):
        up += (edge_points_list[i - 1].x + edge_points_list[i].x - vertices_list[i].x) / (
                division_number_list[i - 1][1] * division_number_list[i][0])
        bottom += 1 / (division_number_list[i - 1][1] * division_number_list[i][0])

    up += (edge_points_list[edge_num - 1].x + edge_points_list[0].x - vertices_list[0].x) / (
            division_number_list[edge_num - 1][1] * division_number_list[0][0])
    bottom += 1 / (division_number_list[edge_num - 1][1] * division_number_list[0][0])

    position_x = up / bottom

    up = 0
    bottom = 0

    for i in range(1, edge_num):
        up += (edge_points_list[i - 1].y + edge_points_list[i].y - vertices_list[i].y) / (
                division_number_list[i - 1][1] * division_number_list[i][0])
        bottom += 1 / (division_number_list[i - 1][1] * division_number_list[i][0])

    up += (edge_points_list[edge_num - 1].y + edge_points_list[0].y - vertices_list[0].y) / (
            division_number_list[edge_num - 1][1] * division_number_list[0][0])
    bottom += 1 / (division_number_list[edge_num - 1][1] * division_number_list[0][0])

    position_y = up / bottom

    position = node(position_x, position_y)

    return position


# 这里是获得某一个face mid point的函数,这里命名与功能其实不符合
def Get_one_radial_point(ver_list, mid_point_list, div_list, edge_num):
    up = 0
    bot = 0
    for i in range(1, edge_num):
        up += (mid_point_list[i - 1].y + mid_point_list[i].y - ver_list[i].y) / (
                div_list[i - 1][1] * div_list[i][0])
        bot += 1 / (div_list[i - 1][1] * div_list[i][0])

    up += (mid_point_list[edge_num - 1].y + mid_point_list[0].y - ver_list[0].y) / (
            div_list[edge_num - 1][1] * div_list[0][0])
    bot += 1 / (div_list[edge_num - 1][1] * div_list[0][0])

    position_y = up / bot

    up = 0
    bot = 0
    for i in range(1, edge_num):
        up += (mid_point_list[i - 1].x + mid_point_list[i].x - ver_list[i].x) / (
                div_list[i - 1][1] * div_list[i][0])
        bot += 1 / (div_list[i - 1][1] * div_list[i][0])

    up += (mid_point_list[edge_num - 1].x + mid_point_list[0].x - ver_list[0].x) / (
            div_list[edge_num - 1][1] * div_list[0][0])
    bot += 1 / (div_list[edge_num - 1][1] * div_list[0][0])
    position_x = up / bot

    face_point_position = node(position_x, position_y)

    return face_point_position


# 这里得到的径向边的结点在当前顶点的顺时针方向下一条边上。这里的after是next的next，考虑是不是要用链表的方式实现。这里
# 实际上是广义的四边形的实现形式，与上面的原理相同。并且由论文可以得到这里的front num[1] == next num [0]
# ver指的是vertex顶点结点，midpoint指的是边的广义的中点
# div_num 的形式是每一条边（u，v）
#
# 这里是获得某条径向边上的所有节点
def Get_one_radial_edge_point(cur_ver, front_ver, next_ver, after_ver, front_div_num, cur_div_num, next_div_num,
                              face_mid_point):
    cur_mid_point = Get_edge_midpoint_position(cur_ver, next_ver, cur_div_num[0], cur_div_num[1])
    next_mid_point = Get_edge_midpoint_position(next_ver, after_ver, next_div_num[0], next_div_num[1])
    front_mid_point = Get_edge_midpoint_position(front_ver, cur_ver, front_div_num[0], front_div_num[1])

    radial_div_num = next_div_num[0]

    a0 = cur_ver
    a1 = next_ver
    a2 = next_mid_point
    a3 = front_mid_point
    ver_list = [a0, a1, a2, a3]

    b0 = cur_mid_point
    b2 = face_mid_point

    radial_point = []

    for i in range(radial_div_num-1):
        b1 = Get_edge_midpoint_position(a2, a1, i + 1, radial_div_num - i - 1)
        b3 = Get_edge_midpoint_position(a3, a0, i + 1, radial_div_num - i - 1)
        mid_point_list = [b0, b1, b2, b3]

        div_list = [cur_div_num, (radial_div_num - i - 1, i + 1), (cur_div_num[1], cur_div_num[0]), (i + 1, radial_div_num - i - 1)]
        temp = Get_one_radial_point(ver_list, mid_point_list, div_list, 4)
        # plt.scatter(temp.x, temp.y, c='y')
        radial_point.append(temp)

        plt.plot([temp.x, b1.x], [temp.y, b1.y], c='r')
        plt.plot([temp.x, b3.x], [temp.y, b3.y], c='r')

    radial_point_num = len(radial_point)

    # 测试所求点是否正确
    # plt.scatter(b2.x, b2.y, s=50, c='b')
    # plt.scatter(b0.x, b0.y, s=50, c='r')

    # 下面是测试绘制面中点和边中点的连线，pass
    # plt.plot([b2.x, b0.x], [b2.y, b0.y], c='r')

    if radial_point_num == 0:
        plt.plot([face_mid_point.x, cur_mid_point.x], [face_mid_point.y, cur_mid_point.y], c='r')
    else:
        plt.plot([face_mid_point.x, radial_point[0].x], [face_mid_point.y, radial_point[0].y], c='r')
        plt.plot([radial_point[radial_point_num-1].x, cur_mid_point.x], [radial_point[radial_point_num-1].y, cur_mid_point.y], c='r')

        for i in range(radial_point_num-1):
            plt.plot([radial_point[i].x, radial_point[i+1].x], [radial_point[i].y, radial_point[i+1].y], c='r')

    return radial_point


def Get_all_radial_points(ver_list, div_num, face_mid_point):
    edge_num = len(ver_list)
    radial_node_list = []

    # 特殊处理第零条边
    cur_ver = ver_list[0]
    front_ver = ver_list[edge_num-1]
    next_ver = ver_list[1]
    after_ver = ver_list[2]
    front_div = div_num[edge_num-1]
    cur_div = div_num[0]
    next_div = div_num[1]
    for i in Get_one_radial_edge_point(cur_ver, front_ver, next_ver, after_ver, front_div, cur_div, next_div,
                                       face_mid_point):
        radial_node_list.append(i)

    for i in range(1, edge_num):
        cur_ver = ver_list[i]
        front_ver = ver_list[i - 1]
        front_div = div_num[i - 1]
        cur_div = div_num[i]

        if i < edge_num - 2:
            next_ver = ver_list[i + 1]
            after_ver = ver_list[i + 2]
            next_div = div_num[i + 1]
        elif i == edge_num - 2:
            next_ver = ver_list[i + 1]
            after_ver = ver_list[0]
            next_div = div_num[i + 1]
        else:
            next_ver = ver_list[0]
            after_ver = ver_list[1]
            next_div = div_num[0]

        for j in Get_one_radial_edge_point(cur_ver, front_ver, next_ver, after_ver, front_div, cur_div, next_div,
                                           face_mid_point):
            radial_node_list.append(j)

    return radial_node_list


# 简单的顶点连线
def plot_outline(ver_list):
    edge_num = len(ver_list)
    plt.plot([ver_list[0].x, ver_list[edge_num-1].x], [ver_list[0].y, ver_list[edge_num-1].y], c='r')
    for i in range(0, edge_num-1):
        plt.plot([ver_list[i].x, ver_list[i+1].x], [ver_list[i].y, ver_list[i+1].y], c='r')

    return 0

# 测试Get_one_radial_point
# pass

# def main_1():
#     v1 = node(0, 0)
#     v2 = node(0, 10)
#     v3 = node(10, 20)
#     v4 = node(10, 0)
#
#     m1 = node(0, 5)
#     m2 = node(5, 15)
#     m3 = node(10, 10)
#     m4 = node(5, 0)
#
#     vertex = [v1, v2, v3, v4]
#     edge_mid = [m1, m2, m3, m4]
#     div_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
#
#     face_mid = Get_one_radial_point(vertex, edge_mid, div_list, 4)
#
#     print(face_mid.x, face_mid.y)
#
#     # 下面是画图
#     for point in vertex:
#         plt.scatter(point.x, point.y, s=5, c='b')
#
#     for point in edge_mid:
#         plt.scatter(point.x, point.y, s=5, c='r')
#
#     plt.scatter(face_mid.x, face_mid.y, s=5, c='g')
#     plt.show()


# 测试中点获取Get_mid_point_from_vertex
# pass
# def main_2():
#     v1 = node(0, 0)
#     v2 = node(0, 10)
#     v3 = node(10, 20)
#     vertex = [v1, v2, v3]
#     div_list = [(1, 1), (1, 2), (2, 1)]
#
#     for point in vertex:
#         plt.scatter(point.x, point.y, s=5, c='b')
#
#     mid_list = Get_mid_point_from_vertex(vertex, div_list)
#     for point in mid_list:
#         plt.scatter(point.x, point.y, s=5, c='r')
#
#
#     plt.show()


# 测试all
def main_3():

    # test_1
    # v1 = node(0, 0)
    # v2 = node(0, 30)
    # v3 = node(25, 20)
    # v4 = node(20, 0)
    #
    # vertex_list = [v1, v2, v3, v4]
    # div_list = [(1, 3), (1, 3), (3, 1), (3, 1)]


    # test_2
    v1 = node(1, 1)
    v2 = node(7, 60)
    v3 = node(20, 30)
    v4 = node(30, -10)

    vertex_list = [v1, v2, v3, v4]
    div_list = [(5, 1), (3, 2), (1, 5), (2, 3)]

    # test3

    # v1 = node(0, 0)
    # v2 = node(20, 10)
    # v3 = node(0, 30)
    # vertex_list = [v1, v2, v3]
    # div_list = [(1, 2), (4, 1), (2, 4)]

    plot_mesh(vertex_list, div_list)


def plot_mesh(vertex_list, div_list):

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    edge_mid_point = Get_mid_point_from_vertex(vertex_list, div_list)
    face_mid_point = Get_face_midpoint_position(vertex_list, edge_mid_point, div_list)

    plot_outline(vertex_list)
    plt.autoscale(False)
    # for point in vertex_list:
        # plt.scatter(point.x, point.y, s=10, c='r')

    # for point in edge_mid_point:
        # plt.scatter(point.x, point.y, s=10, c='b')

    # plt.scatter(face_mid_point.x, face_mid_point.y, s=10, c='g')

    radial_point = Get_all_radial_points(vertex_list, div_list, face_mid_point)

    # for point in radial_point:
    #     plt.scatter(point.x, point.y, s=5, c='y')


if __name__ == "__main__":
    main_3()
    plt.show()
