import numpy as np
import torch
import os
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

# 设置matplotlib后端，支持无图形界面环境
plt.switch_backend('Agg')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_node_features_from_texts(raw_texts, model_name="intfloat/e5-large-v2", batch_size=32):
    logger.info("使用预训练模型生成节点嵌入...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if isinstance(raw_texts[0], torch.Tensor):
        texts = [text.item() if text.numel() == 1 else str(text.tolist()) for text in raw_texts]
    else:
        texts = [str(text) for text in raw_texts]

    input_texts = [f"query: {t}" for t in texts]
    all_embeddings = []

    logger.info("开始生成嵌入...")
    with torch.no_grad():
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

    node_features = torch.cat(all_embeddings, dim=0)
    logger.info(f"生成嵌入维度: {node_features.shape}")
    return node_features

def load_selected_paths(dataset_name: str, path_type: str, data_dir: str = "path_selection") -> Dict:

    file_path = os.path.join(data_dir, f"{dataset_name}_{path_type}_diversity.pt")
    logger.info(f"加载{path_type}多样性路径数据: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None
    
    data = torch.load(file_path, weights_only=False)
    return data

def load_dataset(dataset_name: str) -> Dict:
    dataset_path = f"datasets/{dataset_name}.pt"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
    
    data_dict = torch.load(dataset_path)
    logger.info(f"成功加载 {dataset_name} 数据集")
    return data_dict

def extract_path_nodes(high_paths_data: Dict, low_paths_data: Dict, num_samples: int = 5) -> Tuple[List, List, List, List]:

    high_paths = high_paths_data['paths']
    low_paths = low_paths_data['paths']
    
    # 随机抽取指定数量的路径
    if len(high_paths) > num_samples:
        high_indices = np.random.choice(len(high_paths), num_samples, replace=False)
        selected_high_paths = [high_paths[i] for i in high_indices]
    else:
        selected_high_paths = high_paths
    
    if len(low_paths) > num_samples:
        low_indices = np.random.choice(len(low_paths), num_samples, replace=False)
        selected_low_paths = [low_paths[i] for i in low_indices]
    else:
        selected_low_paths = low_paths
    
    # 提取路径中的所有节点
    high_nodes = set()
    low_nodes = set()
    
    for anchor_id, target_node, path, diversity_score in selected_high_paths:
        high_nodes.update(path)
    
    for anchor_id, target_node, path, diversity_score in selected_low_paths:
        low_nodes.update(path)
    
    high_nodes = list(high_nodes)
    low_nodes = list(low_nodes)
    
    logger.info(f"抽取了 {len(selected_high_paths)} 条高多样性路径，包含 {len(high_nodes)} 个唯一节点")
    logger.info(f"抽取了 {len(selected_low_paths)} 条低多样性路径，包含 {len(low_nodes)} 个唯一节点")
    
    return selected_high_paths, selected_low_paths, high_nodes, low_nodes

def visualize_paths_tsne(data_dict: Dict, 
                        high_nodes: List[int], 
                        low_nodes: List[int], 
                        selected_high_paths: List,
                        selected_low_paths: List,
                        dataset_name: str, 
                        figsize=(15, 12)):

    logger.info(f"\n开始处理 {dataset_name} 数据集路径可视化...")
    
    if 'raw_texts' in data_dict:
        logger.info("使用预训练模型从raw_texts中生成嵌入")
        node_features = generate_node_features_from_texts(data_dict['raw_texts'])
    else:
        raise ValueError("数据集没有'raw_texts'键")
    
    num_nodes = data_dict.get('num_nodes', data_dict['edge_index'].max().item() + 1)
    
    logger.info(f"节点特征维度: {node_features.shape}")
    logger.info(f"总节点数: {num_nodes}")
    
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.cpu().numpy()
    
    logger.info("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(node_features)
    logger.info("t-SNE 降维完成")

    node_types = np.zeros(num_nodes, dtype=int)  # 0: 其他节点, 1: 高多样性节点, 2: 低多样性节点
    node_types[high_nodes] = 1
    node_types[low_nodes] = 2

    plt.figure(figsize=figsize)
    
    other_mask = node_types == 0
    plt.scatter(embeddings_2d[other_mask, 0], embeddings_2d[other_mask, 1],
                c='lightgray', s=8, alpha=0.6, label='其他节点')
    
    high_mask = node_types == 1
    plt.scatter(embeddings_2d[high_mask, 0], embeddings_2d[high_mask, 1],
                c='red', s=50, alpha=0.8, label='高多样性路径节点')
    
    low_mask = node_types == 2
    plt.scatter(embeddings_2d[low_mask, 0], embeddings_2d[low_mask, 1],
                c='blue', s=50, alpha=0.8, label='低多样性路径节点')
    
    logger.info("绘制高多样性路径连接线...")
    for i, (anchor_id, target_node, path, diversity_score) in enumerate(selected_high_paths):
        if len(path) > 1:
            path_coords = embeddings_2d[path]
            plt.plot(path_coords[:, 0], path_coords[:, 1], 
                    color='red', linewidth=2, alpha=0.6, 
                    label='高多样性路径' if i == 0 else "")
            
            plt.scatter(path_coords[0, 0], path_coords[0, 1], 
                       c='darkred', s=100, marker='s', alpha=0.9, 
                       label='高多样性起点' if i == 0 else "")
            plt.scatter(path_coords[-1, 0], path_coords[-1, 1], 
                       c='darkred', s=100, marker='^', alpha=0.9, 
                       label='高多样性终点' if i == 0 else "")
    
    logger.info("绘制低多样性路径连接线...")
    for i, (anchor_id, target_node, path, diversity_score) in enumerate(selected_low_paths):
        if len(path) > 1:
            path_coords = embeddings_2d[path]
            plt.plot(path_coords[:, 0], path_coords[:, 1], 
                    color='blue', linewidth=2, alpha=0.6, 
                    label='低多样性路径' if i == 0 else "")
            
            plt.scatter(path_coords[0, 0], path_coords[0, 1], 
                       c='darkblue', s=100, marker='s', alpha=0.9, 
                       label='低多样性起点' if i == 0 else "")
            plt.scatter(path_coords[-1, 0], path_coords[-1, 1], 
                       c='darkblue', s=100, marker='^', alpha=0.9, 
                       label='低多样性终点' if i == 0 else "")
    
    plt.title(f"{dataset_name.upper()} 数据集路径节点 t-SNE 可视化\n高多样性 vs 低多样性路径", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE 维度 1", fontsize=12)
    plt.ylabel("t-SNE 维度 2", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    high_ratio = len(high_nodes) / num_nodes
    low_ratio = len(low_nodes) / num_nodes
    overlap = len(set(high_nodes) & set(low_nodes))
    
    stats_text = f"""
路径节点统计:
总节点数: {num_nodes}
高多样性节点: {len(high_nodes)} ({high_ratio:.3f})
低多样性节点: {len(low_nodes)} ({low_ratio:.3f})
重叠节点: {overlap}
高多样性路径: {len(selected_high_paths)}条
低多样性路径: {len(selected_low_paths)}条
    """
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    os.makedirs("path_img", exist_ok=True)
    
    save_path = f"path_img/{dataset_name}_paths_tsne.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"路径可视化图片已保存到: {save_path}")
    
    plt.close()

    return embeddings_2d, node_types

def process_dataset_path_visualization(dataset_name: str, num_samples: int = 5):

    logger.info(f"\n开始处理数据集路径可视化: {dataset_name}")
    
    try:
        data_dict = load_dataset(dataset_name)
        
        high_paths_data = load_selected_paths(dataset_name, 'high')
        low_paths_data = load_selected_paths(dataset_name, 'low')
        
        if high_paths_data is None or low_paths_data is None:
            logger.error(f"无法加载数据集 {dataset_name} 的路径数据")
            return
        
        selected_high_paths, selected_low_paths, high_nodes, low_nodes = extract_path_nodes(
            high_paths_data, low_paths_data, num_samples
        )
        
        embeddings_2d, node_types = visualize_paths_tsne(data_dict, high_nodes, low_nodes, selected_high_paths, selected_low_paths, dataset_name)
        
        logger.info(f"\n{dataset_name.upper()} 数据集路径统计:")
        logger.info(f"  - 总节点数: {len(embeddings_2d)}")
        logger.info(f"  - 高多样性节点数: {len(high_nodes)}")
        logger.info(f"  - 低多样性节点数: {len(low_nodes)}")
        logger.info(f"  - 重叠节点数: {len(set(high_nodes) & set(low_nodes))}")
        logger.info(f"  - 前5个高多样性节点: {high_nodes[:5]}")
        logger.info(f"  - 前5个低多样性节点: {low_nodes[:5]}")
        
        logger.info(f"数据集 {dataset_name} 路径可视化完成")
        
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 路径可视化时出错: {str(e)}")
        raise

def main():

    datasets = ["cora", "citeseer"]
    num_samples = 5  # 每种类型抽取5条路径
    
    logger.info("开始路径节点t-SNE可视化")
    logger.info(f"将从每种多样性类型中各抽取 {num_samples} 条路径进行可视化")
    
    for dataset in datasets:
        try:
            process_dataset_path_visualization(dataset, num_samples)
        except Exception as e:
            logger.error(f"处理数据集 {dataset} 路径可视化失败: {str(e)}")
            continue
    
    logger.info("所有数据集路径可视化完成")
    logger.info("可视化结果保存在 path_img 文件夹中")

if __name__ == "__main__":
    main() 