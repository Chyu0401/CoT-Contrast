import torch
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

def load_embeddings_from_hdf5(file_path):
    print(f"从 {file_path} 加载嵌入...")
    
    with h5py.File(file_path, 'r') as f:
        # 假设嵌入存储在'embeddings'键下
        # 如果键名不同，请告诉我实际的键名
        if 'embeddings' in f:
            embeddings = f['embeddings'][:]
        elif 'embeds' in f:
            embeddings = f['embeds'][:]
        else:
            # 列出所有可用的键
            print(f"可用的键: {list(f.keys())}")
            # 使用第一个键
            first_key = list(f.keys())[0]
            embeddings = f[first_key][:]
            print(f"使用键 '{first_key}' 加载嵌入")
    
    # 转换为PyTorch张量
    node_embeddings = torch.FloatTensor(embeddings)
    
    print(f"  嵌入形状: {node_embeddings.shape}")
    return node_embeddings

def rescale_similarity(sim_matrix: torch.Tensor, alpha=3.0):
    """
    使用指数放大和Min-Max归一化来拉大相似度差异
    
    Args:
        sim_matrix: 原始相似度矩阵
        alpha: 指数放大参数，控制差异放大程度 (2~5)
    
    Returns:
        torch.Tensor: 重新缩放的相似度矩阵
    """
    print(f"使用指数放大 (alpha={alpha}) 和Min-Max归一化重新缩放相似度...")
    
    # Step 1: 去掉对角线（自身相似度 = 1）
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(0)
    
    # Step 2: 指数放大小差异
    sim_matrix = sim_matrix.pow(alpha)
    
    # Step 3: Min-Max 归一化到 [0, 1]
    min_val = sim_matrix.min()
    max_val = sim_matrix.max()
    norm_sim = (sim_matrix - min_val) / (max_val - min_val + 1e-8)
    
    # 恢复对角线为1
    norm_sim.fill_diagonal_(1.0)
    
    print(f"  原始相似度范围: [{min_val:.3f}, {max_val:.3f}]")
    print(f"  缩放后相似度范围: [{norm_sim.min():.3f}, {norm_sim.max():.3f}]")
    
    return norm_sim

def compute_node_similarity(v0, v_target, node_embeddings):
    """
    计算两个节点之间的余弦相似度
    
    Args:
        v0: 节点ID
        v_target: 目标节点ID
        node_embeddings: 节点嵌入矩阵 [N x embedding_dim]
    
    Returns:
        float: 余弦相似度得分
    """
    vec1 = node_embeddings[v0].unsqueeze(0)  # [1 x embedding_dim]
    vec2 = node_embeddings[v_target].unsqueeze(0)  # [1 x embedding_dim]
    return F.cosine_similarity(vec1, vec2).item()

def calculate_all_node_similarities(node_embeddings):
    """
    计算所有节点两两之间的相似度
    
    Args:
        node_embeddings: 节点嵌入矩阵 [N x embedding_dim]
    
    Returns:
        torch.Tensor: 相似度矩阵 [N x N]
    """
    print("计算所有节点两两之间的相似度...")
    
    num_nodes = node_embeddings.shape[0]
    print(f"  节点数量: {num_nodes}")
    print(f"  将计算 {num_nodes * (num_nodes - 1) // 2} 个相似度对")
    
    # 初始化相似度矩阵
    similarity_matrix = torch.zeros((num_nodes, num_nodes))
    
    # 计算所有节点对之间的相似度
    for i in tqdm(range(num_nodes), desc="计算相似度"):
        for j in range(i + 1, num_nodes):  # 只计算上三角矩阵，避免重复计算
            sim = compute_node_similarity(i, j, node_embeddings)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # 对称矩阵
    
    # 对角线设为1（节点与自身的相似度）
    similarity_matrix.fill_diagonal_(1.0)
    
    print(f"  相似度矩阵计算完成，形状: {similarity_matrix.shape}")
    return similarity_matrix

def calculate_similarity_for_dataset(dataset_name, alpha=3.0):
    """
    为指定数据集计算节点相似度
    
    Args:
        dataset_name: 数据集名称 ('cora' 或 'citeseer')
        alpha: 指数放大参数
    """
    print(f"开始处理 {dataset_name} 数据集...")
    
    # 文件路径
    embeddings_file = f"embeddings/embeds_{dataset_name}.hdf5"
    
    # 创建输出目录
    output_dir = Path("node_similarity2")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{dataset_name}_node_similarity.pt"
    
    # 检查文件是否存在
    if not os.path.exists(embeddings_file):
        print(f"错误: 嵌入文件 {embeddings_file} 不存在")
        return
    
    # 加载预生成的嵌入
    node_embeddings = load_embeddings_from_hdf5(embeddings_file)
    
    print(f"数据集信息:")
    print(f"  - 节点数量: {node_embeddings.shape[0]}")
    print(f"  - 嵌入维度: {node_embeddings.shape[1]}")
    
    # 计算相似度矩阵
    similarity_matrix = calculate_all_node_similarities(node_embeddings)
    
    # 显示原始相似度统计
    print(f"\n原始相似度统计:")
    print(f"  - 相似度范围: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    print(f"  - 平均相似度: {similarity_matrix.mean():.3f}")
    print(f"  - 标准差: {similarity_matrix.std():.3f}")
    
    # 重新缩放相似度矩阵
    rescaled_similarity_matrix = rescale_similarity(similarity_matrix, alpha=alpha)
    
    # 保存结果
    print(f"保存结果到 {output_file}...")
    torch.save(rescaled_similarity_matrix, output_file)
    
    # 统计信息
    print(f"计算完成!")
    print(f"  - 相似度矩阵形状: {rescaled_similarity_matrix.shape}")
    print(f"  - 缩放后相似度范围: [{rescaled_similarity_matrix.min():.3f}, {rescaled_similarity_matrix.max():.3f}]")
    print(f"  - 缩放后平均相似度: {rescaled_similarity_matrix.mean():.3f}")
    print(f"  - 缩放后标准差: {rescaled_similarity_matrix.std():.3f}")
    
    # 显示一些示例
    print("\n示例相似度 (缩放后):")
    for i in range(min(5, node_embeddings.shape[0])):
        for j in range(i + 1, min(i + 6, node_embeddings.shape[0])):
            print(f"  节点 {i} 与节点 {j}: {rescaled_similarity_matrix[i, j]:.3f}")

def main():
    """
    主函数：为Cora和Citeseer数据集计算节点相似度
    """
    print("=== 使用预生成嵌入的节点相似度计算脚本 (带缩放) ===")
    print()
    
    # 设置指数放大参数
    alpha = 3.0  # 可以调整到2~5之间
    print(f"使用指数放大参数: alpha = {alpha}")
    
    datasets = ['cora', 'citeseer']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        calculate_similarity_for_dataset(dataset, alpha=alpha)
        print(f"{'='*50}")
    
    print("\n所有数据集处理完成!")
    print("结果文件保存在 node_similarity2/ 文件夹下:")
    print("  - cora_node_similarity.pt")
    print("  - citeseer_node_similarity.pt")

if __name__ == "__main__":
    main()
