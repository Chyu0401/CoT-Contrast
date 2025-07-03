import torch
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_node_features_from_texts(raw_texts, max_features=1000):

    print("从文本生成节点特征...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)
    )
    
    features = vectorizer.fit_transform(raw_texts)

    node_features = torch.FloatTensor(features.toarray())
    
    print(f"  生成特征维度: {node_features.shape}")
    return node_features

def compute_node_similarity(v0, v_target, node_features):

    vec1 = node_features[v0].unsqueeze(0)  # [1 x feature_dim]
    vec2 = node_features[v_target].unsqueeze(0)  # [1 x feature_dim]
    return F.cosine_similarity(vec1, vec2).item()

def calculate_all_node_similarities(node_features):
    """
    计算所有节点两两之间的相似度
    
    Args:
        node_features: 节点特征矩阵 [N x feature_dim]
    
    Returns:
        torch.Tensor: 相似度矩阵 [N x N]
    """
    print("计算所有节点两两之间的相似度...")
    
    num_nodes = node_features.shape[0]
    print(f"  节点数量: {num_nodes}")
    print(f"  将计算 {num_nodes * (num_nodes - 1) // 2} 个相似度对")
    
    similarity_matrix = torch.zeros((num_nodes, num_nodes))
    
    # 计算所有节点对之间的相似度
    for i in range(num_nodes):
        if i % 100 == 0:
            print(f"  处理进度: {i}/{num_nodes}")
        
        for j in range(i + 1, num_nodes):  # 只计算上三角矩阵，避免重复计算
            sim = compute_node_similarity(i, j, node_features)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # 对称矩阵
    
    # 对角线设为1（节点与自身的相似度）
    similarity_matrix.fill_diagonal_(1.0)
    
    print(f"  相似度矩阵计算完成，形状: {similarity_matrix.shape}")
    return similarity_matrix

def calculate_similarity_for_dataset(dataset_name):

    print(f"开始处理 {dataset_name} 数据集...")

    data_file = f"datasets/{dataset_name}.pt"

    output_dir = Path("node_similarity")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{dataset_name}_node_similarity.pt"

    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        return

    print("加载数据...")
    data = torch.load(data_file, weights_only=False)
    
    node_features = generate_node_features_from_texts(data['raw_texts'])
    
    print(f"数据集信息:")
    print(f"  - 节点数量: {node_features.shape[0]}")
    print(f"  - 特征维度: {node_features.shape[1]}")

    similarity_matrix = calculate_all_node_similarities(node_features)

    print(f"保存结果到 {output_file}...")
    torch.save(similarity_matrix, output_file)
    
    print(f"计算完成!")
    print(f"  - 相似度矩阵形状: {similarity_matrix.shape}")
    print(f"  - 相似度范围: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    print(f"  - 平均相似度: {similarity_matrix.mean():.3f}")
    
    # 显示一些示例
    print("\n示例相似度:")
    for i in range(min(5, node_features.shape[0])):
        for j in range(i + 1, min(i + 6, node_features.shape[0])):
            print(f"  节点 {i} 与节点 {j}: {similarity_matrix[i, j]:.3f}")

def main():

    datasets = ['cora', 'citeseer']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        calculate_similarity_for_dataset(dataset)
        print(f"{'='*50}")
    
    print("\n所有数据集处理完成!")
    print("结果文件保存在 node_similarity/ 文件夹下:")
    print("  - node_similarity_cora.pt")
    print("  - node_similarity_citeseer.pt")

if __name__ == "__main__":
    main()
