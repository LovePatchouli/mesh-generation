# mesh-evolution
use python for mesh generation , mesh quality and mesh smoothing. simple , stupid but interesting


# 实现Li的‘Quad mesh generation for k-sided faces and hex mesh
# generation for trivalent 1 polyhedra ’中的内容，能够实现对一个k—sided多边形区域的网格划分


singularity_pairs.py
# 实现了Li的‘Multiblock mesh refinement by adding mesh singularities‘中的内容。
# 主要是对一个四边形区域进行添加奇异点对进行划分。求解奇异点位置，径向边上节点的位置，绘制划分结果

store_nodes_as_array.py
# ①重新实现了插入双奇异点的分割算法，实现思路相同，但是把分割得到的节点在每一个block中以矩阵的形式存储。这里感谢Mr.Sun提供思路。
# 这个文件贡献的主要思路也在于此。每个block都可以视为结构化分割，因此block内部用矩阵存储，在每一个block内部就能很轻易地得到节点的拓扑关系。
# block是有限的，因此在block之间的公共边手动把公共节点插入，程序中又更为详细的步骤
# ②实现了laplacian smmothing的经典方法
# ③实现了基于角度优化的方法，将laplacian smoothing 和angle——based的方法先后使用似乎有不错的效果。

pairs_series.py
# 这个实现的是根据一组外部划分数，遍历所有的内部划分数，并对生成结果保存成png文件（不好的选择）
# 注意生成的方案会非常多（视外部分割数而定），建议更改文件保存位置。


2022毕设内容
