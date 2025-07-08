import numpy as np
import torch
import os
import logging
import json
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_path_diversity_data(file_path: str) -> Dict:
    """
    加载路径多样性数据
    
    参数：
    - file_path: str，多样性数据文件路径
    
    返回：
    - Dict，包含多样性矩阵和路径信息的字典
    """
    logger.info(f"加载路径多样性数据: {file_path}")
    
    # 检查文件扩展名
    if file_path.endswith('.json'):
        # 加载JSON格式数据
        with open(file_path, 'r', encoding='utf-8') as f:
            diversity_data = json.load(f)
    else:
        # 加载PT格式数据
        diversity_data = torch.load(file_path, weights_only=False)
    
    return diversity_data

def load_dataset_labels(dataset_name: str) -> np.ndarray:
    """
    加载数据集的节点标签
    
    参数：
    - dataset_name: str，数据集名称
    
    返回：
    - np.ndarray，节点标签数组
    """
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

def extract_all_paths_with_scores(diversity_data: Dict, 
                                 labels: np.ndarray = None) -> List[Tuple[int, int, List[int], float, List[int]]]:
    """
    从路径多样性数据中提取所有路径及其分数和标签
    
    参数：
    - diversity_data: Dict，多样性数据字典
    - labels: np.ndarray，节点标签数组
    
    返回：
    - List[Tuple[int, int, List[int], float, List[int]]]，所有路径的列表，每个元素为(anchor_id, target_node, path, diversity_score, path_labels)
    """
    all_paths = []
    
    # 获取路径数据
    if 'paths_with_diversity' in diversity_data:
        paths_with_diversity = diversity_data['paths_with_diversity']
    else:
        logger.error("数据中没有找到 'paths_with_diversity' 字段")
        return all_paths
    
    for anchor_id, anchor_paths in paths_with_diversity.items():
        anchor_id = int(anchor_id)  # 确保是整数
        
        for path_info in anchor_paths:
            # 新的数据结构
            target_node = path_info['target_node']
            path = path_info['path']
            diversity_score = path_info['diversity_score']
            
            # 获取路径上节点的标签
            path_labels = []
            if labels is not None:
                path_labels = [int(labels[node_id]) for node_id in path]
            
            all_paths.append((anchor_id, target_node, path, diversity_score, path_labels))
    
    logger.info(f"提取了 {len(all_paths)} 条路径")
    return all_paths

def select_paths_by_diversity(all_paths: List[Tuple[int, int, List[int], float, List[int]]], 
                            percentile: float = 20.0) -> Tuple[List[Tuple[int, int, List[int], float, List[int]]], 
                                                             List[Tuple[int, int, List[int], float, List[int]]]]:
    """
    根据多样性分数选择最高和最低百分比的路径
    
    参数：
    - all_paths: List[Tuple[int, int, List[int], float, List[int]]]，所有路径列表
    - percentile: float，要选择的百分比（默认20%）
    
    返回：
    - Tuple[List, List]，(高多样性路径列表, 低多样性路径列表)
    """
    if not all_paths:
        logger.warning("没有找到路径数据")
        return [], []
    
    # 按多样性分数排序
    sorted_paths = sorted(all_paths, key=lambda x: x[3], reverse=True)
    
    # 计算要选择的路径数量
    num_paths = len(sorted_paths)
    num_select = max(1, int(num_paths * percentile / 100.0))
    
    logger.info(f"总路径数: {num_paths}, 选择数量: {num_select} ({percentile}%)")
    
    # 选择最高和最低多样性的路径
    high_diversity_paths = sorted_paths[:num_select]
    low_diversity_paths = sorted_paths[-num_select:]
    
    # 计算统计信息
    high_scores = [path[3] for path in high_diversity_paths]
    low_scores = [path[3] for path in low_diversity_paths]
    
    logger.info(f"高多样性路径 - 数量: {len(high_diversity_paths)}, "
                f"分数范围: [{min(high_scores):.4f}, {max(high_scores):.4f}], "
                f"平均分数: {np.mean(high_scores):.4f}")
    
    logger.info(f"低多样性路径 - 数量: {len(low_diversity_paths)}, "
                f"分数范围: [{min(low_scores):.4f}, {max(low_scores):.4f}], "
                f"平均分数: {np.mean(low_scores):.4f}")
    
    return high_diversity_paths, low_diversity_paths

def save_selected_paths(high_paths: List[Tuple[int, int, List[int], float, List[int]]], 
                       low_paths: List[Tuple[int, int, List[int], float, List[int]]], 
                       dataset_name: str, 
                       output_dir: str = "path_selection"):
    """
    保存选择的高多样性和低多样性路径（PT和JSON格式）
    
    参数：
    - high_paths: List[Tuple[int, int, List[int], float, List[int]]]，高多样性路径列表
    - low_paths: List[Tuple[int, int, List[int], float, List[int]]]，低多样性路径列表
    - dataset_name: str，数据集名称
    - output_dir: str，输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存高多样性路径 - PT格式
    high_diversity_file = os.path.join(output_dir, f"{dataset_name}_high_diversity.pt")
    high_diversity_data = {
        'paths': high_paths,
        'dataset_name': dataset_name,
        'diversity_type': 'high',
        'num_paths': len(high_paths),
        'diversity_scores': [path[3] for path in high_paths]
    }
    torch.save(high_diversity_data, high_diversity_file)
    logger.info(f"高多样性路径已保存(PT): {high_diversity_file}")
    
    # 保存高多样性路径 - JSON格式
    high_diversity_json = os.path.join(output_dir, f"{dataset_name}_high_diversity.json")
    high_json_data = {
        'dataset_name': dataset_name,
        'diversity_type': 'high',
        'num_paths': len(high_paths),
        'diversity_scores': [float(path[3]) for path in high_paths],
        'paths': []
    }
    
    for anchor_id, target_node, path, diversity_score, path_labels in high_paths:
        high_json_data['paths'].append({
            'anchor_id': int(anchor_id),
            'target_node': int(target_node),
            'path': path,
            'diversity_score': float(diversity_score),
            'labels': path_labels
        })
    
    with open(high_diversity_json, 'w', encoding='utf-8') as f:
        json.dump(high_json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"高多样性路径已保存(JSON): {high_diversity_json}")
    
    # 保存低多样性路径 - PT格式
    low_diversity_file = os.path.join(output_dir, f"{dataset_name}_low_diversity.pt")
    low_diversity_data = {
        'paths': low_paths,
        'dataset_name': dataset_name,
        'diversity_type': 'low',
        'num_paths': len(low_paths),
        'diversity_scores': [path[3] for path in low_paths]
    }
    torch.save(low_diversity_data, low_diversity_file)
    logger.info(f"低多样性路径已保存(PT): {low_diversity_file}")
    
    # 保存低多样性路径 - JSON格式
    low_diversity_json = os.path.join(output_dir, f"{dataset_name}_low_diversity.json")
    low_json_data = {
        'dataset_name': dataset_name,
        'diversity_type': 'low',
        'num_paths': len(low_paths),
        'diversity_scores': [float(path[3]) for path in low_paths],
        'paths': []
    }
    
    for anchor_id, target_node, path, diversity_score, path_labels in low_paths:
        low_json_data['paths'].append({
            'anchor_id': int(anchor_id),
            'target_node': int(target_node),
            'path': path,
            'diversity_score': float(diversity_score),
            'labels': path_labels
        })
    
    with open(low_diversity_json, 'w', encoding='utf-8') as f:
        json.dump(low_json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"低多样性路径已保存(JSON): {low_diversity_json}")

def process_dataset(dataset_name: str, percentile: float = 20.0):
    """
    处理单个数据集的路径选择
    
    参数：
    - dataset_name: str，数据集名称
    - percentile: float，要选择的百分比
    """
    logger.info(f"\n开始处理数据集: {dataset_name}")
    
    # 构建文件路径 - 优先使用JSON格式
    diversity_file_json = f"path_diversity/{dataset_name}_path_diversity.json"
    diversity_file_pt = f"path_diversity/{dataset_name}_path_diversity.pt"
    
    # 检查文件是否存在
    if os.path.exists(diversity_file_json):
        diversity_file = diversity_file_json
    elif os.path.exists(diversity_file_pt):
        diversity_file = diversity_file_pt
    else:
        logger.error(f"多样性数据文件不存在: {diversity_file_json} 或 {diversity_file_pt}")
        return
    
    try:
        # 加载多样性数据
        diversity_data = load_path_diversity_data(diversity_file)
        
        # 加载数据集标签
        labels = load_dataset_labels(dataset_name)
        
        # 提取所有路径（包含标签信息）
        all_paths = extract_all_paths_with_scores(diversity_data, labels)
        
        if not all_paths:
            logger.warning(f"数据集 {dataset_name} 没有有效的路径数据")
            return
        
        # 选择高多样性和低多样性路径
        high_diversity_paths, low_diversity_paths = select_paths_by_diversity(all_paths, percentile)
        
        # 保存选择的路径（PT和JSON格式）
        save_selected_paths(high_diversity_paths, low_diversity_paths, dataset_name)
        
        logger.info(f"数据集 {dataset_name} 处理完成")
        
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}")
        raise

def main():
    """
    主函数，处理所有数据集
    """
    datasets = ["cora", "citeseer"]
    percentile = 20.0  # 选择20%的路径
    
    logger.info("开始路径选择处理")
    logger.info(f"将选择每个数据集多样性分数最高和最低的 {percentile}% 路径")
    logger.info("结果将保存为PT和JSON两种格式，包含节点标签信息")
    
    for dataset in datasets:
        try:
            process_dataset(dataset, percentile)
        except Exception as e:
            logger.error(f"处理数据集 {dataset} 失败: {str(e)}")
            continue
    
    logger.info("所有数据集处理完成")
    logger.info("结果保存在 path_selection 文件夹中")

if __name__ == "__main__":
    main()
