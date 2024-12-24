import numpy as np
from calculate_nsm_monte_carlo import compute_nsm
from LLL_reduction import lll_algorithm
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

def closest_lattice_point_enum(B, x, radius=1):
    """
    Find the closest lattice point to x in the lattice defined by generator matrix B
    using pure enumeration.

    :param B: Generator matrix of the lattice (n x n)
    :param x: Target point (n,)
    :param radius: Search radius for integer coordinates (defines the range of enumeration)
    :return: Closest lattice point in the lattice space
    """
    n = B.shape[1]  # Dimension of the lattice
    best_point = None
    min_distance = float('inf')
    
    # Define the search range for integer coordinates
    search_range = range(-radius, radius + 1)
    
    # print(x)
    # print(B)
    # Enumerate all integer coordinate combinations within the range
    for u in product(search_range, repeat=n):
        u = np.array(u)  # Integer coordinate vector
        p = u@B  # Lattice point
        
        
        # Compute the Euclidean distance from x to this lattice point
        distance = np.linalg.norm(x - p)
        
        # Update the closest point if this one is closer
        if distance < min_distance:
            best_point = u
            min_distance = distance
            # print(u)
            # print(p)
            # print(distance)
   
    return best_point



def orthogonalize_and_reduce(B):
    """
    Perform lattice reduction and orthogonalization.
    Orthogonalization is done via Cholesky decomposition to
    convert the Gram matrix to a lower triangular matrix.
    """
    B=lll_algorithm(B)
    # Perform lattice reduction (simplified version using LLL algorithm)
    from scipy.linalg import cholesky
    # Orthogonalize using Cholesky decomposition
    A = B @ B.T  # Compute Gram matrix
    L = cholesky(A, lower=True)  # Cholesky decomposition
    return L


def initialize_lattice(n):
    """
    Generate a random lattice generator matrix B with n dimensions.
    """
    B = np.random.randn(n, n)  # Gaussian random initialization
    B = orthogonalize_and_reduce(B)
    return B
def plot_lattice(B, z, u_hat, id,num_points=5):
    """
    只适合2D
    绘制格子点、随机点 z 和最近格子点 u_hat。
    :param B: 格子生成矩阵
    :param z: 随机点
    :param u_hat: 最近的格子点（整数坐标）
    :param num_points: 每个维度生成的格子点范围
    """
    # 生成格子点的坐标
    lattice_points = []
    for i in range(-num_points, num_points + 1):
        for j in range(-num_points, num_points + 1):
            lattice_points.append(i * B[0] + j * B[1])
    lattice_points = np.array(lattice_points)

    # 将 z 和 u_hat 投影到格子空间

    

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.scatter(lattice_points[:, 0], lattice_points[:, 1], c='k', label='Lattice Points', alpha=0.6)  # 黑点表示格子点
    if z is not None:
        z_proj = z @ B  # 将 z 投影到 B 定义的格子空间
        plt.scatter(z_proj[0], z_proj[1], c='b', label='True Point (z)', s=100)  # 蓝点表示 z
    if u_hat is not None:
        u_hat_proj = u_hat @ B  # 将最近格子点 u_hat 投影到 B 定义的格子空间
        plt.scatter(u_hat_proj[0], u_hat_proj[1], c='r', label='Closest Lattice Point (u_hat)', s=100)  # 红点表示最近点
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lattice Points and Projection of True Point')
    plt.axis('equal')  # 保持轴比例一致
    plt.savefig(f"/home/tangjq/WORK/ML/results/{id}.jpg")
def lattice_quantization_algorithm(n, T=1000000, mu0=0.005, nu=200, Tr=100):
    """
    Stochastic gradient descent for lattice quantization.
    """
    # Initialize generator matrix B
    B = initialize_lattice(n)
    V = np.prod(np.diag(B))  # Compute volume of the Voronoi region
    B = B / V**(1 / n)  # Normalize to unit volume

    # Set up step size decay (annealing scheme)
    mu_t = lambda t: mu0 * (nu**(-t / (T-1)))  # Exponential decay

    for t in tqdm(range(T)):
    
        # Generate a random vector z uniformly in [0, 1)^n
        z = np.random.uniform(0, 1, size=n)

        # Find the closest lattice point and compute residual error
        u_hat = closest_lattice_point_enum(B, z @ B)
        y = z - u_hat  # Residual
        e = y @ B  # Error vector in lattice coordinates
        if t%int(T/10)==0:
            
            plot_lattice(B,z,u_hat,t,5)
 
        for i in range(n):
            for j in range(i):  # Off-diagonal updates
                B[i, j] -= mu_t(t) * y[i] * e[j] 
            # Diagonal updates
            B[i, i] -= mu_t(t) * (y[i] * e[i] - np.linalg.norm(e)**2 / n / B[i, i])
        
        
        # Apply lattice reduction and re-normalize periodically
        if t % Tr == Tr - 1:
            B = orthogonalize_and_reduce(B)
            V = np.prod(np.diag(B))
            B = B / V**(1 / n)
           

    return B

if __name__=="__main__":

    n = 3  # 维度 n=2时NSM真值约为0.08018，n=3时约为0.0785445
    T = 1000000  # 迭代次数
    mu0 = 0.001  # 初始步长
    nu = 200  # 初始与最终步长比
    Tr = 100 # 格子归约周期

    optimized_B = lattice_quantization_algorithm(n, T, mu0, nu, Tr)
    optimized_B/=optimized_B[0,0]
    print("Optimized Generator Matrix:")
    print(optimized_B)



    plot_lattice(optimized_B,None,None,"raw")
    r_B=lll_algorithm(optimized_B)
    print("Optimized Generator Matrix After Reduction:")
    print(r_B)

    print(compute_nsm(optimized_B))
    print(compute_nsm(r_B))

    plot_lattice(r_B,None,None,"after_reduction")
