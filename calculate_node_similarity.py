import torch
import torch.nn.functional as F
import os
from pathlib import Path

def compute_node_similarity(v0, v_target, node_features):
    """
    计算两个节点之间的余弦相似度
    
    Args:
        v0: 锚点节点ID
        v_target: 目标节点ID
        node_features: 节点特征矩阵 [N x feature_dim]
    
    Returns:
        float: 余弦相似度得分
    """
    vec1 = node_features[v0].unsqueeze(0)  # [1 x feature_dim]
    vec2 = node_features[v_target].unsqueeze(0)  # [1 x feature_dim]
    return F.cosine_similarity(vec1, vec2).item()

def calculate_similarity_for_dataset(dataset_name):
    """
    为指定数据集计算节点相似度
    
    Args:
        dataset_name: 数据集名称 ('cora' 或 'citeseer')
    """
    print(f"开始处理 {dataset_name} 数据集...")
    
    # 文件路径
    data_file = f"datasets/{dataset_name}.pt"
    anchors_file = f"anchors/{dataset_name}_anchors.pt"
    hop_matrices_file = f"hop_matrices/hop_matrices_{dataset_name}.pt"
    
    # 创建输出目录
    output_dir = Path("node_similarity")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"node_similarity_{dataset_name}.pt"
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        return
    if not os.path.exists(anchors_file):
        print(f"错误: 锚点文件 {anchors_file} 不存在")
        return
    if not os.path.exists(hop_matrices_file):
        print(f"错误: 跳数矩阵文件 {hop_matrices_file} 不存在")
        return
    
    # 加载数据
    print("加载数据...")
    data = torch.load(data_file, weights_only=False)
    anchors = torch.load(anchors_file, weights_only=False).tolist()
    anchor_hop_matrices = torch.load(hop_matrices_file, weights_only=False)
    
    print(f"数据集信息:")
    print(f"  - 节点数量: {data.x.shape[0]}")
    print(f"  - 特征维度: {data.x.shape[1]}")
    print(f"  - 锚点数量: {len(anchors)}")
    print(f"  - 跳数矩阵形状: {next(iter(anchor_hop_matrices.values())).shape}")
    
    # 获取最大跳数
    max_hop = len(next(iter(anchor_hop_matrices.values())))
    print(f"  - 最大跳数: {max_hop}")
    
    # 计算相似度
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
                sim = compute_node_similarity(anchor, target, data.x)
                anchor_result.append((target, d + 1, sim))  # d+1 表示真实跳数
        
        similarity_dict[anchor] = anchor_result
    
    # 保存结果
    print(f"保存结果到 {output_file}...")
    torch.save(similarity_dict, output_file)
    
    # 统计信息
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
    """
    主函数：为Cora和Citeseer数据集计算节点相似度
    """
    print("=== 节点相似度计算脚本 ===")
    print()
    
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
