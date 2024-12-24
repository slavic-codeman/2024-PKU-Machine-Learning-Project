# 2024 PKU Machine Learning Project

## Goal
复现论文Optimization and Identification of Lattice Quantizers.pdf的结果。

## Environment
没有特别要求，基本上是Python+Numpy+Matplotlib+Scipy。

## Progress
目前的进展：
### calculate_nsm_monte_carlo.py
使用蒙特卡洛积分估计任意一个Lattice的Normalized Second Momentum。

### LLL_reduction.py
对Generator Matrix矩阵的LLL归约，使得基向量更短且更正交。时间复杂度为O(n^4)。

### sgd_calculate.py
对论文里SGD求解n维最优Lattice(NSM最小)算法的复现。这是主要运行的文件。
## Tasks
1. 查找一下各个n维度下最优Lattice的NSM值，原论文里已经有了n较大情况的，补充一下较低维度的数据。 
2. 蒙特卡洛积分估计NSM速度很慢，特别是对于n较大的情况，能否改进。
3. sgd_calculate.py中的closest point方法（找到空间中任意一个点的最近格点对应的坐标）采用枚举法，世间复杂度O(3^n)，能否改进。
4. 论文中使用Theta Series可视化Lattice进行比较，能否实现。


