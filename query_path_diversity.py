import torch
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_path_diversity_data(file_path: str):
    """
    加载路径多样性数据
    
    参数：
    - file_path: str，路径多样性数据文件路径
    
    返回：
    - Dict，包含多样性矩阵和路径信息的字典
    """
    logger.info(f"加载路径多样性数据: {file_path}")
    diversity_data = torch.load(file_path, weights_only=False)
    return diversity_data

def query_path_diversity_by_anchor_id(diversity_data, anchor_id: int, path_index: int = 0):
    """
    通过锚点ID和路径索引查询路径多样性
    
    参数：
    - diversity_data: Dict，路径多样性数据
    - anchor_id: int，锚点ID
    - path_index: int，路径索引（默认为0，即第一条路径）
    
    返回：
    - Tuple[int, List[int], float]，(target_node, path, diversity_score)
    """
    paths_with_diversity = diversity_data['paths_with_diversity']
    
    if anchor_id not in paths_with_diversity:
        logger.error(f"锚点 {anchor_id} 不存在")
        return None
    
    anchor_paths = paths_with_diversity[anchor_id]
    
    if path_index >= len(anchor_paths):
        logger.error(f"路径索引 {path_index} 超出范围，该锚点只有 {len(anchor_paths)} 条路径")
        return None
    
    target_node, path, diversity_score = anchor_paths[path_index]
    return target_node, path, diversity_score

def query_path_diversity_by_matrix_index(diversity_data, anchor_idx: int, path_idx: int):
    """
    通过矩阵索引查询路径多样性分数
    
    参数：
    - diversity_data: Dict，路径多样性数据
    - anchor_idx: int，锚点在矩阵中的索引
    - path_idx: int，路径在矩阵中的索引
    
    返回：
    - float，多样性分数
    """
    diversity_matrix = diversity_data['diversity_matrix']
    
    if anchor_idx >= diversity_matrix.shape[0]:
        logger.error(f"锚点索引 {anchor_idx} 超出范围")
        return None
    
    if path_idx >= diversity_matrix.shape[1]:
        logger.error(f"路径索引 {path_idx} 超出范围")
        return None
    
    return diversity_matrix[anchor_idx, path_idx]

def get_anchor_id_from_index(diversity_data, anchor_idx: int):
    """
    从矩阵索引获取锚点ID
    
    参数：
    - diversity_data: Dict，路径多样性数据
    - anchor_idx: int，锚点在矩阵中的索引
    
    返回：
    - int，锚点ID
    """
    anchor_ids = sorted(diversity_data['paths_with_diversity'].keys())
    
    if anchor_idx >= len(anchor_ids):
        logger.error(f"锚点索引 {anchor_idx} 超出范围")
        return None
    
    return anchor_ids[anchor_idx]

def demo_query_functions(dataset_name: str = "cora"):
    """
    演示各种查询功能
    """
    logger.info(f"演示 {dataset_name} 数据集的路径多样性查询功能")
    
    # 加载数据
    file_path = f"path_diversity/{dataset_name}_path_diversity.pt"
    try:
        diversity_data = load_path_diversity_data(file_path)
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        return
    
    # 显示基本信息
    logger.info(f"数据集: {diversity_data['dataset_name']}")
    logger.info(f"矩阵形状: {diversity_data['matrix_shape']}")
    logger.info(f"锚点数量: {diversity_data['num_anchors']}")
    
    # 获取锚点ID列表
    anchor_ids = sorted(diversity_data['paths_with_diversity'].keys())
    logger.info(f"锚点ID列表: {anchor_ids[:10]}{'...' if len(anchor_ids) > 10 else ''}")
    
    # 示例1：通过锚点ID查询第一条路径
    if anchor_ids:
        anchor_id = anchor_ids[0]
        result = query_path_diversity_by_anchor_id(diversity_data, anchor_id, 0)
        if result:
            target_node, path, diversity_score = result
            logger.info(f"\n示例1 - 锚点 {anchor_id} 的第一条路径:")
            logger.info(f"  目标节点: {target_node}")
            logger.info(f"  路径: {path}")
            logger.info(f"  多样性分数: {diversity_score:.4f}")
    
    # 示例2：通过矩阵索引查询
    diversity_score = query_path_diversity_by_matrix_index(diversity_data, 0, 1)
    if diversity_score is not None:
        anchor_id = get_anchor_id_from_index(diversity_data, 0)
        logger.info(f"\n示例2 - 矩阵索引 [0,1] 的多样性分数:")
        logger.info(f"  对应锚点ID: {anchor_id}")
        logger.info(f"  多样性分数: {diversity_score:.4f}")
    
    # 示例3：查询所有锚点的第一条路径多样性分数
    logger.info(f"\n示例3 - 所有锚点的第一条路径多样性分数:")
    for i, anchor_id in enumerate(anchor_ids[:5]):  # 只显示前5个
        result = query_path_diversity_by_anchor_id(diversity_data, anchor_id, 0)
        if result:
            target_node, path, diversity_score = result
            logger.info(f"  锚点 {anchor_id}: {diversity_score:.4f}")
    
    # 示例4：统计信息
    diversity_matrix = diversity_data['diversity_matrix']
    non_zero_scores = diversity_matrix[diversity_matrix > 0]
    
    logger.info(f"\n示例4 - 多样性分数统计:")
    logger.info(f"  总分数数量: {len(non_zero_scores)}")
    logger.info(f"  平均分数: {np.mean(non_zero_scores):.4f}")
    logger.info(f"  最高分数: {np.max(non_zero_scores):.4f}")
    logger.info(f"  最低分数: {np.min(non_zero_scores):.4f}")

if __name__ == "__main__":
    # 演示cora数据集
    demo_query_functions("cora")
    
    print("\n" + "="*60)
    
    # 演示citeseer数据集
    demo_query_functions("citeseer") 