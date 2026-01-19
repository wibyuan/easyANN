import networkx as nx
import sympy
import sys

# 增加递归深度，虽然这里主要靠矩阵运算
sys.setrecursionlimit(20000)

class ExactResearcher:
    def construct_koch_like_graph(self, iterations):
        """
        构造逻辑：
        G1: 三角形 (3边, 3点)
        G_n: 在 G_{n-1} 的所有【外部边界边】上，加一个点连成新三角形，保留原边。
        """
        G = nx.Graph()
        # G1: 0-1, 1-2, 2-0
        edges = [(0, 1), (1, 2), (2, 0)]
        G.add_edges_from(edges)
        
        # 维护边界边集合
        boundary_edges = set(edges)
        next_node_idx = 3
        
        for i in range(iterations - 1):
            new_boundary = set()
            for u, v in list(boundary_edges):
                # 确保边还在图里（防御性编程）
                if not G.has_edge(u, v):
                    continue
                    
                w = next_node_idx
                next_node_idx += 1
                
                # 加新边 (u,w), (w,v)
                G.add_edge(u, w)
                G.add_edge(w, v)
                
                # 原边 (u,v) 变为内部边，保留
                # 新边成为下一轮的边界
                new_boundary.add((u, w))
                new_boundary.add((w, v))
            
            boundary_edges = new_boundary
            
        return G

    def get_exact_spanning_trees(self, G):
        """
        使用 sympy 计算拉普拉斯矩阵的精确行列式。
        这利用了 Python 的任意精度整数特性，结果绝对准确。
        """
        # 获取拉普拉斯矩阵 (稀疏格式转稠密)
        L = nx.laplacian_matrix(G).todense()
        
        # 转换为 sympy 的 Matrix 对象，确保使用整数运算
        # 我们只需要由 n-1 阶主子式
        L_reduced = L[:-1, :-1] # 删掉最后一行一列
        M = sympy.Matrix(L_reduced)
        
        # 计算行列式 (Gaussian elimination with exact fractions/integers)
        # method='berkowitz' 对于大矩阵通常比较稳定，或者默认
        det = M.det()
        
        return det

    def factorize_integer(self, n):
        """
        简单的质因数分解展示工具
        """
        factors = sympy.factorint(n)
        result = []
        for p in sorted(factors.keys()):
            exp = factors[p]
            if exp == 1:
                result.append(f"{p}")
            else:
                result.append(f"{p}^{exp}")
        return " * ".join(result)

# --- 主程序 ---

researcher = ExactResearcher()

print("=== 精确生成树数量 (Exact Spanning Tree Counts) ===")
print("正在计算，请稍候（G5/G6 矩阵较大，sympy 需要一点时间）...\n")

# 建议先跑前5代，G6 (96节点) 的符号行列式可能需要几分钟
for n in range(1, 10):
    G = researcher.construct_koch_like_graph(n)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # 计算精确值
    trees = researcher.get_exact_spanning_trees(G)
    
    # 质因数分解
    factors = researcher.factorize_integer(trees)
    
    print(f"[G_{n}]")
    print(f"节点: {num_nodes}, 边: {num_edges}")
    print(f"生成树数量 (Exact): {trees}")
    print(f"质因数分解: {factors}")
    print("-" * 40)

print("\n提示：")
print("1. 检查 G4, G5 的因子，看是否有 2, 3, 5, 7 以外的质数。")
print("2. 观察指数的变化规律。")