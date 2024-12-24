import numpy as np
import matplotlib.pyplot as plt

def plot_lattice(B,num_points=5):
    """
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
    plt.scatter(B[0,0], B[0,1], c='b', label='V1', s=100)  # 蓝点表示 z
    plt.scatter(B[1,0], B[1,1], c='r', label='V2', s=100)  # 红点表示最近点
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lattice Points and Projection of True Point')
    plt.axis('equal')  # 保持轴比例一致
    plt.savefig(f"lattice.jpg")

def gram_schmidt(basis):
    """
    Gram-Schmidt process to compute the orthogonal basis.
    Returns the orthogonal basis and the coefficients µ.
    """
    n = len(basis)
    d = len(basis[0])
    ortho_basis = np.zeros((n, d))
    mu = np.zeros((n, n))
    
    for i in range(n):
        ortho_basis[i] = basis[i]
        for j in range(i):
            mu[i, j] = np.dot(basis[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j])
            ortho_basis[i] -= mu[i, j] * ortho_basis[j]
    
    return ortho_basis, mu

def lll_algorithm(basis, delta=0.75):
    """
    Lenstra–Lenstra–Lovász (LLL) algorithm implementation in Python.
    Input:
        basis: 2D list or numpy array representing the basis vectors.
        delta: Lovász condition constant (default is 0.75).
    Output:
        Reduced basis as a numpy array.
    """
    n = len(basis)
    basis = np.array(basis, dtype=float)
    k = 1  # Start with the second vector (index 1)

    while k < n:
        # Perform Gram-Schmidt orthogonalization
        ortho_basis, mu = gram_schmidt(basis)

        # Size reduction: adjust the k-th basis vector
        for j in range(k - 1, -1, -1):
            basis[k] -= np.round(mu[k, j]) * basis[j]
        
        # Update Gram-Schmidt orthogonalization after size reduction
        ortho_basis, mu = gram_schmidt(basis)

        # Lovász condition
        if np.linalg.norm(ortho_basis[k])**2 >= (delta - mu[k, k - 1]**2) * np.linalg.norm(ortho_basis[k - 1])**2:
            # If Lovász condition is satisfied, move to the next vector
            k += 1
        else:
            # Swap basis[k] and basis[k - 1]
            basis[[k, k - 1]] = basis[[k - 1, k]]
            # Ensure k doesn't go below 1
            k = max(k - 1, 1)
    
    return basis

# Example usage

if __name__=="__main__":

    # 示例使用
    A = np.array([
        [np.sqrt(3), 1],
        [np.sqrt(3)/3,1]
    ], dtype=float)

    print("原始格基矩阵:")
    print(A)

    A_reduced = lll_algorithm(A.copy())

    print("\n简化后的格基矩阵:")
    print(A_reduced)

    plot_lattice(A_reduced)
