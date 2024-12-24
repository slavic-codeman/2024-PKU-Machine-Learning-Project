import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from scipy.integrate import dblquad

def compute_distance_square_integral(basis_vectors):
    """
    计算0点的Voronoi区域内到原点的欧式距离平方的积分。
    
    参数:
    - basis_vectors: n个线性无关的n维向量 (numpy数组, 形状为 (n, n))
    
    返回:
    - 积分值
    """
    # 生成晶格的Voronoi区域
    dimension = basis_vectors.shape[0]
    
    # 定义一个网格，生成包含原点的晶格点
    max_extent = 1  # 假设只生成第一壳的晶格点
    lattice_points = np.array([
        sum((c-1) * basis_vectors[i] for i, c in enumerate(coords))
        for coords in np.ndindex(*([3] * dimension))
    ])

    vor = Voronoi(lattice_points)
  
    # 找到包含原点的Voronoi区域顶点
    region_index = vor.point_region[np.where(np.all(vor.points == 0, axis=1))[0][0]]
    region_vertices = vor.vertices[vor.regions[region_index]]
   
    # 确保区域是凸的，并创建凸包
    hull = ConvexHull(region_vertices)
    
    # 定义需要积分的函数
    def distance_squared(*coords):
        point = np.array(coords)
        return np.dot(point, point)  # 欧几里得距离的平方

    # 定义区域边界（凸包内的点）
    def in_hull(point, hull):
        """
        判断点是否在凸包内
        """
        return all(
            np.dot(eq[:-1], point) + eq[-1] <= 1e-8
            for eq in hull.equations
        )
    
    # 使用蒙特卡洛方法进行积分
    num_samples = 200000  # 样本点数量
    bounding_box_min = np.min(region_vertices, axis=0)
    bounding_box_max = np.max(region_vertices, axis=0)
    samples = np.random.uniform(bounding_box_min, bounding_box_max, size=(num_samples, dimension))
    
    # 过滤在凸包内的点
    in_region_samples = samples[np.array([in_hull(sample, hull) for sample in samples])]
    
    # 计算距离平方
    distances_squared = np.sum(in_region_samples**2, axis=1)
    
    # 计算体积
    region_volume = hull.volume
    
    # 计算积分值
    integral = np.mean(distances_squared) * region_volume
    
    return integral
def compute_nsm(basis_vectors):
    n=basis_vectors.shape[0]
    A=basis_vectors@basis_vectors.T
    V=np.sqrt(np.linalg.det(A))
    second_momentum = compute_distance_square_integral(basis_vectors)
    nsm=second_momentum/(V**(1+2/n)*n)
    return nsm
if __name__=="__main__":
    # 示例: 2维情况下
    basis_vectors = np.array([[0, 1        ],
 [np.sqrt(3)/2,1/2]])
    n=basis_vectors.shape[0]
    A=basis_vectors@basis_vectors.T
    V=np.sqrt(np.linalg.det(A))
    second_momentum = compute_distance_square_integral(basis_vectors)
    nsm=second_momentum/(V**(1+2/n)*n)
    print("NSM=",nsm)
