import os
import json
import torch
import numpy as np
from openai import OpenAI
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置API客户端
os.environ["DASHSCOPE_API_KEY"] = "sk-096d9d2e3aac472aad1d3c0d34c3f29f"
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def load_json(file_path: str) -> Dict:
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file_name: str, data: Dict):
    """保存JSON文件"""
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_dataset_data(dataset_name: str) -> Dict:
    """加载数据集数据"""
    dataset_file = f"datasets/{dataset_name}.pt"
    if not os.path.exists(dataset_file):
        logger.error(f"数据集文件不存在: {dataset_file}")
        return {}
    
    logger.info(f"加载数据集: {dataset_file}")
    dataset_data = torch.load(dataset_file, weights_only=False)
    
    if 'raw_texts' not in dataset_data:
        logger.error(f"数据集 {dataset_name} 不包含 'raw_texts' 键")
        return {}
    
    logger.info(f"成功加载数据集 {dataset_name}，包含 {len(dataset_data['raw_texts'])} 个节点")
    return dataset_data

def load_node_similarity(dataset_name: str) -> np.ndarray:
    """加载节点相似度矩阵"""
    similarity_file = f"node_similarity/{dataset_name}_node_similarity.pt"
    if not os.path.exists(similarity_file):
        logger.error(f"节点相似度文件不存在: {similarity_file}")
        return None
    
    logger.info(f"加载节点相似度矩阵: {similarity_file}")
    similarity_matrix = torch.load(similarity_file, weights_only=False)
    return similarity_matrix.numpy()

def get_node_texts(dataset_data: Dict, path: List[int]) -> List[str]:
    """获取路径中节点的文本"""
    node_texts = []
    for node_id in path:
        if node_id < len(dataset_data['raw_texts']):
            text = dataset_data['raw_texts'][node_id]
            # 处理可能的tensor格式
            if isinstance(text, torch.Tensor):
                if text.numel() == 1:
                    text = str(text.item())
                else:
                    text = str(text.tolist())
            node_texts.append(str(text))
        else:
            node_texts.append(f"Node_{node_id}")
    return node_texts

def calculate_node_similarities_for_path(path: List[int], similarity_matrix: np.ndarray) -> List[float]:
    """计算路径中相邻节点间的相似度"""
    similarities = []
    for i in range(len(path) - 1):
        node1, node2 = path[i], path[i + 1]
        if node1 < similarity_matrix.shape[0] and node2 < similarity_matrix.shape[1]:
            sim = similarity_matrix[node1, node2]
            similarities.append(float(sim))
        else:
            similarities.append(0.0)
    return similarities

def find_max_difference_position(similarities: List[float]) -> int:
    """找到相似度差异最大的位置"""
    if len(similarities) <= 1:
        return 0
    
    # 计算相邻相似度的差异
    differences = []
    for i in range(len(similarities) - 1):
        diff = abs(similarities[i] - similarities[i + 1])
        differences.append(diff)
    
    # 找到差异最大的位置
    max_diff_idx = np.argmax(differences)
    return max_diff_idx + 1  # 返回分割点位置

def split_path_symmetrically(path: List[int]) -> Tuple[List[int], List[int]]:
    """对称分割路径"""
    mid = len(path) // 2
    first_half = path[:mid]
    second_half = path[mid:]
    return first_half, second_half

def split_path_at_difference(path: List[int], similarities: List[float]) -> Tuple[List[int], List[int]]:
    """从差异最大处分割路径"""
    split_pos = find_max_difference_position(similarities)
    first_half = path[:split_pos]
    second_half = path[split_pos:]
    return first_half, second_half

def split_explanation_symmetrically(explanation: str) -> Tuple[str, str]:
    """对称分割解释文本"""
    sentences = explanation.split('. ')
    mid = len(sentences) // 2
    
    first_half = '. '.join(sentences[:mid])
    second_half = '. '.join(sentences[mid:])
    
    # 确保句子完整性
    if not first_half.endswith('.'):
        first_half += '.'
    if not second_half.endswith('.'):
        second_half += '.'
    
    return first_half, second_half

def split_explanation_at_difference(explanation: str, split_pos: int, path_length: int) -> Tuple[str, str]:
    """根据路径分割位置分割解释文本"""
    sentences = explanation.split('. ')
    
    # 根据分割位置比例确定文本分割点
    split_ratio = split_pos / path_length
    text_split_pos = int(len(sentences) * split_ratio)
    
    first_half = '. '.join(sentences[:text_split_pos])
    second_half = '. '.join(sentences[text_split_pos:])
    
    # 确保句子完整性
    if not first_half.endswith('.'):
        first_half += '.'
    if not second_half.endswith('.'):
        second_half += '.'
    
    return first_half, second_half

def generate_split_decision(path: List[int], diversity_score: float, node_texts: List[str], 
                           labels: List[int], similarities: List[float]) -> Dict[str, Any]:
    """使用大模型决定如何分割路径"""
    
    # 构建输入信息
    path_info = f"路径: {path}"
    diversity_info = f"多样性分数: {diversity_score:.6f}"
    texts_info = f"节点文本: {node_texts}"
    labels_info = f"节点标签: {labels}"
    similarities_info = f"节点相似度: {[f'{s:.3f}' for s in similarities]}"
    
    prompt = f"""请分析以下路径信息并决定如何分割：

路径信息：
{path_info}
{diversity_info}
{texts_info}
{labels_info}
{similarities_info}

分割规则：
- 如果多样性分数较高（>0.002），使用对称分割
- 如果多样性分数较低（≤0.002），从差异最大节点处分割

请返回JSON格式的分割决策：
{{
    "split_method": "symmetric" 或 "difference",
    "split_position": 分割位置（整数，从0开始）,
    "reasoning": "分割理由"
}}"""

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        response = completion.choices[0].message.content
        
        # 解析JSON响应
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
            return decision
        else:
            # 如果无法解析JSON，使用默认逻辑
            if diversity_score > 0.002:
                return {
                    "split_method": "symmetric",
                    "split_position": len(path) // 2,
                    "reasoning": "高多样性路径使用对称分割"
                }
            else:
                split_pos = find_max_difference_position(similarities)
                return {
                    "split_method": "difference",
                    "split_position": split_pos,
                    "reasoning": "低多样性路径从差异最大处分割"
                }
    except Exception as e:
        logger.error(f"大模型调用失败: {e}")
        # 使用默认逻辑
        if diversity_score > 0.002:
            return {
                "split_method": "symmetric",
                "split_position": len(path) // 2,
                "reasoning": "高多样性路径使用对称分割"
            }
        else:
            split_pos = find_max_difference_position(similarities)
            return {
                "split_method": "difference",
                "split_position": split_pos,
                "reasoning": "低多样性路径从差异最大处分割"
            }

def process_path_data(path_data: Dict, dataset_data: Dict, similarity_matrix: np.ndarray) -> Dict[str, Any]:
    """处理单条路径数据"""
    path = path_data['path']
    diversity_score = path_data['diversity_score']
    explanation = path_data['explanation']
    labels = path_data['labels']
    
    # 获取节点文本
    node_texts = get_node_texts(dataset_data, path)
    
    # 计算节点相似度
    similarities = calculate_node_similarities_for_path(path, similarity_matrix)
    
    # 使用大模型决定分割方法
    split_decision = generate_split_decision(path, diversity_score, node_texts, labels, similarities)
    
    # 根据决策进行分割
    if split_decision['split_method'] == 'symmetric':
        # 对称分割
        path_first, path_second = split_path_symmetrically(path)
        explanation_first, explanation_second = split_explanation_symmetrically(explanation)
        split_pos = len(path) // 2
    else:
        # 从差异最大处分割
        split_pos = split_decision['split_position']
        path_first, path_second = split_path_at_difference(path, similarities)
        explanation_first, explanation_second = split_explanation_at_difference(
            explanation, split_pos, len(path)
        )
    
    # 获取分割后路径的节点文本和标签
    node_texts_first = get_node_texts(dataset_data, path_first)
    node_texts_second = get_node_texts(dataset_data, path_second)
    labels_first = labels[:split_pos] if split_pos < len(labels) else labels
    labels_second = labels[split_pos:] if split_pos < len(labels) else []
    
    return {
        'original_path': path,
        'original_explanation': explanation,
        'original_diversity_score': diversity_score,
        'original_labels': labels,
        'original_node_texts': node_texts,
        'original_similarities': similarities,
        'split_decision': split_decision,
        'split_position': split_pos,
        'path_first': path_first,
        'path_second': path_second,
        'explanation_first': explanation_first,
        'explanation_second': explanation_second,
        'labels_first': labels_first,
        'labels_second': labels_second,
        'node_texts_first': node_texts_first,
        'node_texts_second': node_texts_second
    }

def process_explanation_file(explanation_file: str, dataset_name: str, output_dir: str = "contrastive_data"):
    """处理解释文件中的所有路径"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    logger.info(f"加载解释文件: {explanation_file}")
    explanation_data = load_json(explanation_file)
    
    logger.info(f"加载数据集: {dataset_name}")
    dataset_data = load_dataset_data(dataset_name)
    
    logger.info(f"加载节点相似度矩阵: {dataset_name}")
    similarity_matrix = load_node_similarity(dataset_name)
    
    if not dataset_data or similarity_matrix is None:
        logger.error("数据加载失败")
        return
    
    # 提取数据集名称和多样性类型
    dataset = explanation_data.get('dataset', dataset_name)
    diversity_type = explanation_data.get('diversity_type', 'unknown')
    
    # 处理路径数据
    results = []
    paths_data = explanation_data.get('results', [])
    
    logger.info(f"开始处理 {len(paths_data)} 条路径")
    
    for i, path_data in enumerate(tqdm(paths_data, desc="处理路径")):
        try:
            processed_data = process_path_data(path_data, dataset_data, similarity_matrix)
            results.append(processed_data)
            
            # 每处理10条保存一次进度
            if (i + 1) % 10 == 0:
                temp_output_file = os.path.join(output_dir, f"temp_{dataset}_{diversity_type}_split_data.json")
                save_json(temp_output_file, {
                    'dataset': dataset,
                    'diversity_type': diversity_type,
                    'processed_count': len(results),
                    'results': results
                })
                logger.info(f"已保存临时进度: {len(results)} 条")
                
        except Exception as e:
            logger.error(f"处理路径 {i} 时出错: {e}")
            continue
    
    # 保存最终结果
    output_file = os.path.join(output_dir, f"{dataset}_{diversity_type}_split_data.json")
    final_data = {
        'dataset': dataset,
        'diversity_type': diversity_type,
        'total_paths': len(paths_data),
        'processed_count': len(results),
        'results': results
    }
    
    save_json(output_file, final_data)
    logger.info(f"处理完成！结果保存到: {output_file}")
    logger.info(f"成功处理: {len(results)}/{len(paths_data)} 条路径")

def main():
    """主函数"""
    # 处理explanation文件夹中的所有文件
    explanation_dir = "explanation"
    
    if not os.path.exists(explanation_dir):
        logger.error(f"解释文件夹不存在: {explanation_dir}")
        return
    
    # 获取所有解释文件
    explanation_files = [f for f in os.listdir(explanation_dir) if f.endswith('.json')]
    
    if not explanation_files:
        logger.error(f"在 {explanation_dir} 中没有找到JSON文件")
        return
    
    logger.info(f"找到 {len(explanation_files)} 个解释文件")
    
    for explanation_file in explanation_files:
        file_path = os.path.join(explanation_dir, explanation_file)
        logger.info(f"\n处理文件: {explanation_file}")
        
        # 从文件名推断数据集名称
        if 'cora' in explanation_file:
            dataset_name = 'cora'
        elif 'citeseer' in explanation_file:
            dataset_name = 'citeseer'
        else:
            logger.warning(f"无法从文件名推断数据集名称: {explanation_file}")
            continue
        
        try:
            process_explanation_file(file_path, dataset_name)
        except Exception as e:
            logger.error(f"处理文件 {explanation_file} 时出错: {e}")
            continue
    
    logger.info("所有文件处理完成！")

if __name__ == "__main__":
    main()
