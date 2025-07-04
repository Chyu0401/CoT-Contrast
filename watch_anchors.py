import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import os
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

# 设置matplotlib后端，支持无图形界面环境
plt.switch_backend('Agg')

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_node_features_from_texts(raw_texts, model_name="intfloat/e5-large-v2", batch_size=32):
    """
    使用预训练模型生成节点嵌入
    """
    print("使用预训练模型生成节点嵌入...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 确保文本是字符串格式
    if isinstance(raw_texts[0], torch.Tensor):
        texts = [text.item() if text.numel() == 1 else str(text.tolist()) for text in raw_texts]
    else:
        texts = [str(text) for text in raw_texts]

    input_texts = [f"query: {t}" for t in texts]
    all_embeddings = []

    print("开始生成嵌入...")
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
    print(f"生成嵌入维度: {node_features.shape}")
    return node_features

def load_dataset_and_anchors(dataset_name):
    dataset_path = f"datasets/{dataset_name}.pt"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
    
    data_dict = torch.load(dataset_path)
    print(f"成功加载 {dataset_name} 数据集")
    
    anchors_path = f"anchors/{dataset_name}_anchors.json"
    if not os.path.exists(anchors_path):
        raise FileNotFoundError(f"找不到锚点文件: {anchors_path}")
    
    with open(anchors_path, 'r') as f:
        anchors = json.load(f)
    print(f"成功加载 {dataset_name} 锚点，数量: {len(anchors)}")
    
    return data_dict, anchors

def visualize_anchors_tsne(data_dict, anchors, dataset_name, figsize=(12, 10)):
    print(f"\n开始处理 {dataset_name} 数据集...")
    
    # 检查是否有节点特征，如果没有则从raw_texts提取
    if 'x' in data_dict:
        node_features = data_dict['x']
        print("使用预定义的节点特征")
    elif 'raw_texts' in data_dict:
        print("使用预训练模型从raw_texts中生成嵌入")
        node_features = generate_node_features_from_texts(data_dict['raw_texts'])
    else:
        raise ValueError("数据集中既没有'x'也没有'raw_texts'键")
    
    num_nodes = data_dict.get('num_nodes', data_dict['edge_index'].max().item() + 1)
    
    print(f"节点特征维度: {node_features.shape}")
    print(f"总节点数: {num_nodes}")
    
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.cpu().numpy()
    
    print("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(node_features)
    print("t-SNE 降维完成")

    is_anchor = np.zeros(num_nodes, dtype=bool)
    is_anchor[anchors] = True

    plt.figure(figsize=figsize)
    
    plt.scatter(embeddings_2d[~is_anchor, 0], embeddings_2d[~is_anchor, 1],
                c='lightgray', s=8, alpha=0.6, label='其他节点')
    
    plt.scatter(embeddings_2d[is_anchor, 0], embeddings_2d[is_anchor, 1],
                c='red', s=50, alpha=0.8, label='锚点节点')
    
    plt.title(f"{dataset_name.upper()} 数据集节点 t-SNE 可视化 (锚点高亮)\n使用预训练模型嵌入", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE 维度 1", fontsize=12)
    plt.ylabel("t-SNE 维度 2", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    anchor_ratio = len(anchors) / num_nodes
    plt.text(0.02, 0.98, f'总节点数: {num_nodes}\n锚点数量: {len(anchors)}\n锚点比例: {anchor_ratio:.3f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 创建 graph_img 文件夹
    os.makedirs("graph_img", exist_ok=True)
    
    # 保存图片到 graph_img 文件夹
    save_path = f"graph_img/{dataset_name}_anchors_tsne.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
    
    # 关闭图形以释放内存
    plt.close()
    
    return embeddings_2d, is_anchor

def main():
    datasets = ['cora', 'citeseer']
    
    for dataset_name in datasets:
        try:
            print(f"\n{'='*40}")
            print(f"处理数据集: {dataset_name.upper()}")
            print(f"{'='*40}")
            
            data_dict, anchors = load_dataset_and_anchors(dataset_name)
            
            embeddings_2d, is_anchor = visualize_anchors_tsne(data_dict, anchors, dataset_name)
            
            print(f"\n{dataset_name.upper()} 数据集统计:")
            print(f"  - 总节点数: {len(embeddings_2d)}")
            print(f"  - 锚点数量: {len(anchors)}")
            print(f"  - 锚点比例: {len(anchors)/len(embeddings_2d):.3f}")
            print(f"  - 前10个锚点ID: {anchors[:10]}")
            
        except Exception as e:
            print(f"处理 {dataset_name} 数据集时出错: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("所有数据集处理完成!")
    print("图片已保存到 graph_img/ 文件夹")
    print("=" * 60)

if __name__ == "__main__":
    main() 