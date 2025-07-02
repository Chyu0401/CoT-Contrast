import torch
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_node_features_from_texts(raw_texts, max_features=1000):

    print("从文本生成节点特征...")
    
    # 使用TF-IDF向量化文本
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

def calculate_similarity_for_dataset(dataset_name):

    print(f"开始处理 {dataset_name} 数据集...")
    

    data_file = f"datasets/{dataset_name}.pt"
    anchors_file = f"anchors/{dataset_name}_anchors.pt"
    hop_matrices_file = f"hop_matrices/hop_matrices_{dataset_name}.pt"
    

    output_dir = Path("node_similarity")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"node_similarity_{dataset_name}.pt"
    

    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        return
    if not os.path.exists(anchors_file):
        print(f"错误: 锚点文件 {anchors_file} 不存在")
        return
    if not os.path.exists(hop_matrices_file):
        print(f"错误: 跳数矩阵文件 {hop_matrices_file} 不存在")
        return
    

    print("加载数据...")
    data = torch.load(data_file, weights_only=False)
    anchors = torch.load(anchors_file, weights_only=False).tolist()
    anchor_hop_matrices = torch.load(hop_matrices_file, weights_only=False)
    
    # 从文本生成节点特征
    node_features = generate_node_features_from_texts(data['raw_texts'])
    
    print(f"数据集信息:")
    print(f"  - 节点数量: {node_features.shape[0]}")
    print(f"  - 特征维度: {node_features.shape[1]}")
    print(f"  - 锚点数量: {len(anchors)}")
    print(f"  - 跳数矩阵形状: {next(iter(anchor_hop_matrices.values())).shape}")
    

    max_hop = len(next(iter(anchor_hop_matrices.values())))
    print(f"  - 最大跳数: {max_hop}")
    

    print("开始计算节点相似度...")
    similarity_dict = {}
    
    for i, anchor in enumerate(anchors):
        if i % 10 == 0:
            print(f"  处理锚点 {i+1}/{len(anchors)}: {anchor}")
        
        anchor_result = []
        
        for d in range(max_hop):  # d = 0, 1, ..., max_hop-1
            hop_vec = anchor_hop_matrices[anchor][d]
            target_nodes = torch.where(torch.tensor(hop_vec) == 1)[0].tolist()
            
            for target in target_nodes:
                sim = compute_node_similarity(anchor, target, node_features)
                anchor_result.append((target, d + 1, sim))  # d+1 表示真实跳数
        
        similarity_dict[anchor] = anchor_result
    

    print(f"保存结果到 {output_file}...")
    torch.save(similarity_dict, output_file)
    

    total_pairs = sum(len(pairs) for pairs in similarity_dict.values())
    print(f"计算完成!")
    print(f"  - 总相似度对数量: {total_pairs}")
    print(f"  - 平均每个锚点的相似度对数量: {total_pairs / len(anchors):.1f}")
    
    # 显示一些示例
    print("\n示例结果:")
    for anchor in list(similarity_dict.keys())[:2]:  # 显示前2个锚点的结果
        pairs = similarity_dict[anchor][:5]  # 显示前5个相似度对
        print(f"  锚点 {anchor}: {pairs}")

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
