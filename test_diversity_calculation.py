import numpy as np
import warnings

def path_diversity_score_original(path_nodes, similarity_matrix):
    """原始版本（有警告）"""
    N = len(path_nodes)
    if N <= 1:
        return 0.0

    sub_sim = similarity_matrix[np.ix_(path_nodes, path_nodes)]
    np.fill_diagonal(sub_sim, 0)

    row_sums = sub_sim.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    P = np.zeros_like(sub_sim)
    P[~zero_rows] = sub_sim[~zero_rows] / row_sums[~zero_rows]

    # 这里会产生警告
    log_P = np.where(P > 0, np.log(P), 0)
    h = -np.sum(P * log_P, axis=1)

    max_entropy = np.log(N - 1) if N > 1 else 0
    h[zero_rows] = max_entropy

    diversity = np.mean(max_entropy - h)
    return diversity

def path_diversity_score_fixed(path_nodes, similarity_matrix):
    """修复版本（无警告）"""
    N = len(path_nodes)
    if N <= 1:
        return 0.0

    sub_sim = similarity_matrix[np.ix_(path_nodes, path_nodes)]
    np.fill_diagonal(sub_sim, 0)

    row_sums = sub_sim.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    P = np.zeros_like(sub_sim)
    P[~zero_rows] = sub_sim[~zero_rows] / row_sums[~zero_rows]

    # 更安全的方式
    h = np.zeros(N)
    for i in range(N):
        if not zero_rows[i]:
            non_zero_mask = P[i] > 0
            if np.any(non_zero_mask):
                h[i] = -np.sum(P[i][non_zero_mask] * np.log(P[i][non_zero_mask]))

    max_entropy = np.log(N - 1) if N > 1 else 0
    h[zero_rows] = max_entropy

    diversity = np.mean(max_entropy - h)
    return diversity

def test_equivalence():
    """测试两个版本的结果是否相同"""
    print("测试多样性计算函数的等价性...")
    
    # 测试用例1：简单路径
    path = [0, 2, 3]
    sim_mat = np.array([
        [1.0, 0.1, 0.8, 0.3, 0.0],
        [0.1, 1.0, 0.2, 0.4, 0.5],
        [0.8, 0.2, 1.0, 0.9, 0.1],
        [0.3, 0.4, 0.9, 1.0, 0.2],
        [0.0, 0.5, 0.1, 0.2, 1.0],
    ])
    
    # 抑制警告进行测试
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result1 = path_diversity_score_original(path, sim_mat)
    
    result2 = path_diversity_score_fixed(path, sim_mat)
    
    print(f"原始版本结果: {result1:.6f}")
    print(f"修复版本结果: {result2:.6f}")
    print(f"结果是否相同: {np.abs(result1 - result2) < 1e-10}")
    
    # 测试用例2：包含零相似度的路径
    path2 = [0, 1, 4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result3 = path_diversity_score_original(path2, sim_mat)
    
    result4 = path_diversity_score_fixed(path2, sim_mat)
    
    print(f"\n测试用例2:")
    print(f"原始版本结果: {result3:.6f}")
    print(f"修复版本结果: {result4:.6f}")
    print(f"结果是否相同: {np.abs(result3 - result4) < 1e-10}")
    
    return np.abs(result1 - result2) < 1e-10 and np.abs(result3 - result4) < 1e-10

if __name__ == "__main__":
    is_equivalent = test_equivalence()
    if is_equivalent:
        print("\n✅ 两个版本的结果完全一致，警告不影响计算正确性")
    else:
        print("\n❌ 两个版本的结果不一致") 