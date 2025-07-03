import numpy as np
import torch
import os
from typing import List, Dict, Tuple
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def path_diversity_score(path_nodes: List[int], similarity_matrix: np.ndarray) -> float:
    
    N = len(path_nodes)
    if N <= 1:
        return 0.0  # 单节点路径多样性无意义

    # 1. 提取路径节点对应的相似度子矩阵 (N x N)
    # similarity_matrix[i,j] 是节点 i 和节点 j 的相似度
    sub_sim = similarity_matrix[np.ix_(path_nodes, path_nodes)]  # 选中路径节点行列

    # 2. 对角线（节点与自身）不参与熵计算，设为0
    np.fill_diagonal(sub_sim, 0)

    # 3. 对每一行（每个节点A），归一化得到概率分布 P_An
    # 避免除以0，先求行和
    row_sums = sub_sim.sum(axis=1, keepdims=True)
    # 行和为0的行（孤立节点）暂时赋予均匀分布或跳过
    zero_rows = (row_sums == 0).flatten()
    P = np.zeros_like(sub_sim)
    P[~zero_rows] = sub_sim[~zero_rows] / row_sums[~zero_rows]

    # 4. 计算每个节点的熵 h_A = -sum P_An log P_An
    log_P = np.where(P > 0, np.log(P), 0)
    h = -np.sum(P * log_P, axis=1)  # shape: (N, )

    # 对于零行（孤立节点），熵设为最大熵 log(N-1)
    max_entropy = np.log(N - 1) if N > 1 else 0
    h[zero_rows] = max_entropy

    # 5. 计算路径多样性，平均"反熵"
    diversity = np.mean(max_entropy - h)

    return diversity

def load_shortest_paths(file_path: str) -> Dict[int, List[Tuple[int, List[int]]]]:
    """
    加载最短路径数据
    
    参数：
    - file_path: str，最短路径文件路径
    
    返回：
    - Dict[int, List[Tuple[int, List[int]]]]，锚点ID到(target_node, path)元组列表的映射
    """
    logger.info(f"加载最短路径数据: {file_path}")
    shortest_paths = torch.load(file_path, weights_only=False)
    return shortest_paths

def load_node_similarity(file_path: str) -> np.ndarray:
    logger.info(f"加载节点相似度矩阵: {file_path}")
    similarity_matrix = torch.load(file_path, weights_only=False)
    return similarity_matrix.numpy()

def calculate_path_diversity_matrix(shortest_paths: Dict[int, List[Tuple[int, List[int]]]], 
                                  similarity_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, List[int], float]]]]:
    """
    计算所有路径的多样性分数矩阵
    
    参数：
    - shortest_paths: Dict[int, List[Tuple[int, List[int]]]]，锚点ID到(target_node, path)元组列表的映射
    - similarity_matrix: np.ndarray，节点相似度矩阵
    
    返回：
    - np.ndarray，路径多样性分数矩阵，shape = [num_anchors, max_path_length]
    - Dict[int, List[Tuple[int, List[int], float]]]，锚点ID到(target_node, path, diversity_score)元组列表的映射
    """
    num_anchors = len(shortest_paths)
    max_path_length = max(len(paths) for paths in shortest_paths.values()) if shortest_paths else 0
    
    logger.info(f"计算路径多样性矩阵: {num_anchors} 个锚点, 最大路径长度 {max_path_length}")
    
    diversity_matrix = np.zeros((num_anchors, max_path_length))
    
    paths_with_diversity = {}
    
    # 获取锚点ID列表并排序
    anchor_ids = sorted(shortest_paths.keys())
    
    for i, anchor_id in enumerate(anchor_ids):
        anchor_paths = shortest_paths[anchor_id]  # List[Tuple[int, List[int]]]
        logger.info(f"处理锚点 {anchor_id} ({i+1}/{num_anchors}): {len(anchor_paths)} 条路径")
        
        anchor_paths_with_diversity = []
        
        for j, (target_node, path) in enumerate(anchor_paths):
            if j >= max_path_length:
                break
            diversity_score = path_diversity_score(path, similarity_matrix)
            diversity_matrix[i, j] = diversity_score
            
            # 将路径信息和多样性分数一起保存
            anchor_paths_with_diversity.append((target_node, path, diversity_score))
        
        paths_with_diversity[anchor_id] = anchor_paths_with_diversity
    
    return diversity_matrix, paths_with_diversity

def save_path_diversity_matrix(diversity_matrix: np.ndarray, 
                             paths_with_diversity: Dict[int, List[Tuple[int, List[int], float]]],
                             output_path: str, 
                             dataset_name: str):

    logger.info(f"保存路径多样性数据到: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建包含所有信息的字典
    diversity_data = {
        'diversity_matrix': diversity_matrix,
        'paths_with_diversity': paths_with_diversity,
        'matrix_shape': diversity_matrix.shape,
        'num_anchors': len(paths_with_diversity),
        'dataset_name': dataset_name
    }
    
    # 保存为.pt格式
    torch.save(diversity_data, output_path)
    
    logger.info(f"数据集 {dataset_name} 路径多样性数据已保存")
    logger.info(f"矩阵形状: {diversity_matrix.shape}")
    logger.info(f"多样性分数统计: 均值={np.mean(diversity_matrix):.4f}, "
                f"标准差={np.std(diversity_matrix):.4f}, "
                f"最小值={np.min(diversity_matrix):.4f}, "
                f"最大值={np.max(diversity_matrix):.4f}")

def process_dataset(dataset_name: str):

    logger.info(f"\n开始处理数据集: {dataset_name}")
    
    shortest_paths_file = f"shortest_paths/{dataset_name}_shortest_paths.pt"
    node_similarity_file = f"node_similarity/{dataset_name}_node_similarity.pt"
    output_file = f"path_diversity/{dataset_name}_path_diversity.pt"
    
    if not os.path.exists(shortest_paths_file):
        logger.error(f"最短路径文件不存在: {shortest_paths_file}")
        return
    
    if not os.path.exists(node_similarity_file):
        logger.error(f"节点相似度文件不存在: {node_similarity_file}")
        return
    

    shortest_paths = load_shortest_paths(shortest_paths_file)
    similarity_matrix = load_node_similarity(node_similarity_file)
    
    diversity_matrix, paths_with_diversity = calculate_path_diversity_matrix(shortest_paths, similarity_matrix)
    
    save_path_diversity_matrix(diversity_matrix, paths_with_diversity, output_file, dataset_name)

def load_path_diversity_data(file_path: str) -> Dict:

    logger.info(f"加载路径多样性数据: {file_path}")
    diversity_data = torch.load(file_path, weights_only=False)
    return diversity_data

def get_path_diversity_example():

    logger.info("演示路径多样性分数查询...")
    
    diversity_data = load_path_diversity_data("path_diversity/cora_path_diversity.pt")
    
    diversity_matrix = diversity_data['diversity_matrix']
    paths_with_diversity = diversity_data['paths_with_diversity']
    
    anchor_idx = 0  # 第一个锚点
    path_idx = 1    # 第二条路径
    diversity_score = diversity_matrix[anchor_idx, path_idx]
    logger.info(f"锚点索引{anchor_idx}的第{path_idx}条路径多样性分数: {diversity_score:.4f}")
    
    anchor_ids = sorted(paths_with_diversity.keys())
    if anchor_ids:
        anchor_id = anchor_ids[0]  # 第一个锚点ID
        anchor_paths = paths_with_diversity[anchor_id]
        
        if len(anchor_paths) > 1:
            target_node, path, diversity_score = anchor_paths[1]  # 第二条路径
            logger.info(f"锚点{anchor_id}到目标{target_node}的路径: {path}")
            logger.info(f"该路径的多样性分数: {diversity_score:.4f}")
    
    return diversity_data

def main():

    datasets = ["cora", "citeseer"]
    
    for dataset in datasets:
        try:
            process_dataset(dataset)
            logger.info(f"数据集 {dataset} 处理完成")
        except Exception as e:
            logger.error(f"处理数据集 {dataset} 时出错: {str(e)}")
    
    logger.info("所有数据集处理完成")
    
    # # 演示如何查询路径多样性分数
    # try:
    #     logger.info("\n" + "="*50)
    #     logger.info("演示路径多样性查询功能")
    #     logger.info("="*50)
    #     get_path_diversity_example()
    # except Exception as e:
    #     logger.warning(f"演示查询功能时出错: {str(e)}")

if __name__ == "__main__":
    main()
