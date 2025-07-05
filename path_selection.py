import numpy as np
import torch
import os
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_path_diversity_data(file_path: str) -> Dict:
 
    logger.info(f"加载路径多样性数据: {file_path}")
    diversity_data = torch.load(file_path, weights_only=False)
    return diversity_data

def extract_all_paths_with_scores(paths_with_diversity: Dict[int, List[Tuple[int, List[int], float]]]) -> List[Tuple[int, int, List[int], float]]:

    all_paths = []
    
    for anchor_id, anchor_paths in paths_with_diversity.items():
        for target_node, path, diversity_score in anchor_paths:
            all_paths.append((anchor_id, target_node, path, diversity_score))
    
    logger.info(f"提取了 {len(all_paths)} 条路径")
    return all_paths

def select_paths_by_diversity(all_paths: List[Tuple[int, int, List[int], float]], 
                            percentile: float = 20.0) -> Tuple[List[Tuple[int, int, List[int], float]], 
                                                             List[Tuple[int, int, List[int], float]]]:

    if not all_paths:
        logger.warning("没有找到路径数据")
        return [], []
    
    # 按多样性分数排序
    sorted_paths = sorted(all_paths, key=lambda x: x[3], reverse=True)
    
    num_paths = len(sorted_paths)
    num_select = max(1, int(num_paths * percentile / 100.0))
    
    logger.info(f"总路径数: {num_paths}, 选择数量: {num_select} ({percentile}%)")
    
    high_diversity_paths = sorted_paths[:num_select]
    low_diversity_paths = sorted_paths[-num_select:]
    
    high_scores = [path[3] for path in high_diversity_paths]
    low_scores = [path[3] for path in low_diversity_paths]
    
    logger.info(f"高多样性路径 - 数量: {len(high_diversity_paths)}, "
                f"分数范围: [{min(high_scores):.4f}, {max(high_scores):.4f}], "
                f"平均分数: {np.mean(high_scores):.4f}")
    
    logger.info(f"低多样性路径 - 数量: {len(low_diversity_paths)}, "
                f"分数范围: [{min(low_scores):.4f}, {max(low_scores):.4f}], "
                f"平均分数: {np.mean(low_scores):.4f}")
    
    return high_diversity_paths, low_diversity_paths

def save_selected_paths(high_paths: List[Tuple[int, int, List[int], float]], 
                       low_paths: List[Tuple[int, int, List[int], float]], 
                       dataset_name: str, 
                       output_dir: str = "path_selection"):

    os.makedirs(output_dir, exist_ok=True)
    
    high_diversity_file = os.path.join(output_dir, f"{dataset_name}_high_diversity.pt")
    high_diversity_data = {
        'paths': high_paths,
        'dataset_name': dataset_name,
        'diversity_type': 'high',
        'num_paths': len(high_paths),
        'diversity_scores': [path[3] for path in high_paths]
    }
    torch.save(high_diversity_data, high_diversity_file)
    logger.info(f"高多样性路径已保存: {high_diversity_file}")
    
    low_diversity_file = os.path.join(output_dir, f"{dataset_name}_low_diversity.pt")
    low_diversity_data = {
        'paths': low_paths,
        'dataset_name': dataset_name,
        'diversity_type': 'low',
        'num_paths': len(low_paths),
        'diversity_scores': [path[3] for path in low_paths]
    }
    torch.save(low_diversity_data, low_diversity_file)
    logger.info(f"低多样性路径已保存: {low_diversity_file}")

def process_dataset(dataset_name: str, percentile: float = 20.0):

    logger.info(f"\n开始处理数据集: {dataset_name}")
    
    diversity_file = f"path_diversity/{dataset_name}_path_diversity.pt"
    
    if not os.path.exists(diversity_file):
        logger.error(f"多样性数据文件不存在: {diversity_file}")
        return
    
    try:
        diversity_data = load_path_diversity_data(diversity_file)
        paths_with_diversity = diversity_data['paths_with_diversity']
        
        all_paths = extract_all_paths_with_scores(paths_with_diversity)
        
        if not all_paths:
            logger.warning(f"数据集 {dataset_name} 没有有效的路径数据")
            return
        
        high_diversity_paths, low_diversity_paths = select_paths_by_diversity(all_paths, percentile)
        
        save_selected_paths(high_diversity_paths, low_diversity_paths, dataset_name)
        
        logger.info(f"数据集 {dataset_name} 处理完成")
        
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}")
        raise

def main():
    datasets = ["cora", "citeseer"]
    percentile = 20.0 
    
    logger.info("开始路径选择处理")
    logger.info(f"将选择每个数据集多样性分数最高和最低的 {percentile}% 路径")
    
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
