import numpy as np
import torch
import torch.nn.functional as F
import os
import json
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def path_diversity_score(path_nodes: List[int], similarity_matrix: np.ndarray) -> float:
    
    N = len(path_nodes)
    if N <= 1:
        return 0.0 

    sub_sim = similarity_matrix[np.ix_(path_nodes, path_nodes)]
    
    sim_tensor = torch.from_numpy(sub_sim).float()
    sim_tensor.fill_diagonal_(0.0)
    sim_tensor = torch.clamp(sim_tensor, min=0.0)
    
    # 归一化相似度：p'_ik = p_ik / sum p_ik
    row_sums = sim_tensor.sum(dim=1, keepdim=True) + 1e-12
    probs = sim_tensor / row_sums 
    
    # 熵计算：h_i = -sum_k p'_ik log(p'_ik)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1) 
    
    max_entropy = torch.log(torch.tensor(N - 1.0, dtype=probs.dtype))
    diversity_score = (max_entropy - entropy).mean().item()

    return diversity_score

def load_shortest_paths(file_path: str) -> Dict[int, List[Tuple[int, List[int]]]]:

    logger.info(f"加载最短路径数据: {file_path}")
    shortest_paths = torch.load(file_path, weights_only=False)
    return shortest_paths

def load_node_similarity(file_path: str) -> np.ndarray:
    logger.info(f"加载节点相似度矩阵: {file_path}")
    similarity_matrix = torch.load(file_path, weights_only=False)
    return similarity_matrix.numpy()

def load_dataset_labels(dataset_name: str) -> np.ndarray:
    """加载数据集的节点标签"""
    dataset_file = f"datasets/{dataset_name}.pt"
    logger.info(f"加载数据集标签: {dataset_file}")
    
    if not os.path.exists(dataset_file):
        logger.error(f"数据集文件不存在: {dataset_file}")
        return None
    
    dataset = torch.load(dataset_file, weights_only=False)
    if 'y' in dataset:
        labels = dataset['y']
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        logger.info(f"成功加载标签，标签数量: {len(labels)}")
        return labels
    else:
        logger.error(f"数据集中没有找到标签字段 'y'")
        return None

def calculate_path_diversity_matrix(shortest_paths: Dict[int, List[Tuple[int, List[int]]]], 
                                  similarity_matrix: np.ndarray,
                                  labels: np.ndarray = None) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, List[int], float, List[int]]]]]:
    """
    - shortest_paths: Dict[int, List[Tuple[int, List[int]]]]，锚点ID到(target_node, path)元组列表的映射
    - similarity_matrix: np.ndarray，节点相似度矩阵
    - labels: np.ndarray，节点标签数组
    
    返回：
    - np.ndarray，路径多样性分数矩阵，shape = [num_anchors, max_path_length]
    - Dict[int, List[Tuple[int, List[int], float, List[int]]]]，锚点ID到(target_node, path, diversity_score, path_labels)元组列表的映射
    """
    num_anchors = len(shortest_paths)
    max_path_length = max(len(paths) for paths in shortest_paths.values()) if shortest_paths else 0
    
    logger.info(f"计算路径多样性矩阵: {num_anchors} 个锚点, 最大路径长度 {max_path_length}")
    
    diversity_matrix = np.zeros((num_anchors, max_path_length))
    
    paths_with_diversity = {}
    
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
            
            # 获取路径上节点的标签
            path_labels = []
            if labels is not None:
                path_labels = [int(labels[node_id]) for node_id in path]
            
            anchor_paths_with_diversity.append((target_node, path, diversity_score, path_labels))
        
        paths_with_diversity[anchor_id] = anchor_paths_with_diversity
    
    return diversity_matrix, paths_with_diversity

def save_path_diversity_matrix(diversity_matrix: np.ndarray, 
                             paths_with_diversity: Dict[int, List[Tuple[int, List[int], float, List[int]]]],
                             output_path: str, 
                             dataset_name: str):

    logger.info(f"保存路径多样性数据到: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 准备PT格式数据
    diversity_data = {
        'paths_with_diversity': paths_with_diversity,
        'num_anchors': len(paths_with_diversity),
        'dataset_name': dataset_name
    }
    
    # 保存PT格式
    torch.save(diversity_data, output_path)
    
    # 准备JSON格式数据
    json_data = {
        'dataset_name': dataset_name,
        'num_anchors': len(paths_with_diversity),
        'paths_with_diversity': {}
    }
    
    # 转换路径数据为JSON可序列化格式
    for anchor_id, paths in paths_with_diversity.items():
        json_data['paths_with_diversity'][str(anchor_id)] = []
        for target_node, path, diversity_score, path_labels in paths:
            json_data['paths_with_diversity'][str(anchor_id)].append({
                'target_node': int(target_node),
                'path': path,
                'diversity_score': float(diversity_score),
                'labels': path_labels
            })
    
    # 保存JSON格式
    json_output_path = output_path.replace('.pt', '.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"数据集 {dataset_name} 路径多样性数据已保存")
    logger.info(f"PT格式: {output_path}")
    logger.info(f"JSON格式: {json_output_path}")
    logger.info(f"锚点数量: {len(paths_with_diversity)}")
    
    # 计算多样性分数统计
    all_scores = []
    for paths in paths_with_diversity.values():
        for _, _, diversity_score, _ in paths:
            all_scores.append(diversity_score)
    
    if all_scores:
        all_scores = np.array(all_scores)
        logger.info(f"多样性分数统计: 均值={np.mean(all_scores):.4f}, "
                    f"标准差={np.std(all_scores):.4f}, "
                    f"最小值={np.min(all_scores):.4f}, "
                    f"最大值={np.max(all_scores):.4f}")

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
    
    # 加载标签数据
    labels = load_dataset_labels(dataset_name)

    shortest_paths = load_shortest_paths(shortest_paths_file)
    similarity_matrix = load_node_similarity(node_similarity_file)
    
    diversity_matrix, paths_with_diversity = calculate_path_diversity_matrix(shortest_paths, similarity_matrix, labels)
    
    save_path_diversity_matrix(diversity_matrix, paths_with_diversity, output_file, dataset_name)

def load_path_diversity_data(file_path: str) -> Dict:

    logger.info(f"加载路径多样性数据: {file_path}")
    diversity_data = torch.load(file_path, weights_only=False)
    return diversity_data

def load_path_diversity_data_json(file_path: str) -> Dict:
    """从JSON文件加载路径多样性数据"""
    logger.info(f"从JSON加载路径多样性数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        diversity_data = json.load(f)
    return diversity_data

def get_path_diversity_example():

    logger.info("演示路径多样性分数查询...")
    
    # 尝试加载PT格式
    pt_file = "path_diversity/cora_path_diversity.pt"
    json_file = "path_diversity/cora_path_diversity.json"
    
    if os.path.exists(pt_file):
        diversity_data = load_path_diversity_data(pt_file)
        paths_with_diversity = diversity_data['paths_with_diversity']
        
        anchor_ids = sorted(paths_with_diversity.keys())
        if anchor_ids:
            anchor_id = anchor_ids[0]  # 第一个锚点ID
            anchor_paths = paths_with_diversity[anchor_id]
            
            if len(anchor_paths) > 1:
                target_node, path, diversity_score, path_labels = anchor_paths[1]  # 第二条路径
                logger.info(f"锚点{anchor_id}到目标{target_node}的路径: {path}")
                logger.info(f"该路径的多样性分数: {diversity_score:.4f}")
                logger.info(f"该路径的节点标签: {path_labels}")
    
    # 尝试加载JSON格式
    if os.path.exists(json_file):
        logger.info("\n演示JSON格式数据查询...")
        json_data = load_path_diversity_data_json(json_file)
        
        # 获取第一个锚点的第一条路径信息
        anchor_ids = list(json_data['paths_with_diversity'].keys())
        if anchor_ids:
            anchor_id = anchor_ids[0]
            anchor_paths = json_data['paths_with_diversity'][anchor_id]
            
            if anchor_paths:
                path_info = anchor_paths[0]
                logger.info(f"JSON格式 - 锚点{anchor_id}到目标{path_info['target_node']}的路径: {path_info['path']}")
                logger.info(f"JSON格式 - 该路径的多样性分数: {path_info['diversity_score']:.4f}")
                logger.info(f"JSON格式 - 该路径的节点标签: {path_info['labels']}")
    
    return diversity_data if 'diversity_data' in locals() else None

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
