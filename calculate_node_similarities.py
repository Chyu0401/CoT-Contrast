import torch
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_node_features_from_texts(raw_texts, model_name="intfloat/e5-large-v2", batch_size=32):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    input_texts = [f"query: {t}" for t in raw_texts]
    all_embeddings = []

    print("开始生成嵌入...")
    with torch.no_grad():
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

    node_features = torch.cat(all_embeddings, dim=0)
    print(f"  生成特征维度: {node_features.shape}")
    return node_features

def compute_node_similarity(v0, v_target, node_features):
    vec1 = node_features[v0].unsqueeze(0)
    vec2 = node_features[v_target].unsqueeze(0)
    return F.cosine_similarity(vec1, vec2).item()

def calculate_all_node_similarities(node_features):

    num_nodes = node_features.shape[0]
    print(f"  节点数量: {num_nodes}")
    print(f"  将计算 {num_nodes * (num_nodes - 1) // 2} 个相似度对")

    similarity_matrix = torch.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        if i % 100 == 0:
            print(f"  处理进度: {i}/{num_nodes}")
        for j in range(i + 1, num_nodes):
            sim = compute_node_similarity(i, j, node_features)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    similarity_matrix.fill_diagonal_(1.0)
    print(f"  相似度矩阵计算完成，形状: {similarity_matrix.shape}")
    return similarity_matrix

def calculate_similarity_for_dataset(dataset_name, model_name="intfloat/e5-large-v2"):
    print(f"开始处理 {dataset_name} 数据集...")

    data_file = f"datasets/{dataset_name}.pt"
    output_dir = Path("node_similarity")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{dataset_name}_node_similarity.pt"

    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        return

    print("加载数据...")
    data = torch.load(data_file, weights_only=False)

    node_features = generate_node_features_from_texts(data['raw_texts'], model_name=model_name)

    print(f"数据集信息:")
    print(f"  - 节点数量: {node_features.shape[0]}")
    print(f"  - 特征维度: {node_features.shape[1]}")

    similarity_matrix = calculate_all_node_similarities(node_features)

    print(f"保存结果到 {output_file}...")
    torch.save(similarity_matrix, output_file)

    print(f"计算完成!")
    print(f"  - 相似度矩阵形状: {similarity_matrix.shape}")
    print(f"  - 相似度范围: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    print(f"  - 平均相似度: {similarity_matrix.mean():.3f}")

    print("\n示例相似度:")
    for i in range(min(5, node_features.shape[0])):
        for j in range(i + 1, min(i + 6, node_features.shape[0])):
            print(f"  节点 {i} 与节点 {j}: {similarity_matrix[i, j]:.3f}")

def main():
    datasets = ['cora', 'citeseer']
    model_name = "intfloat/e5-large-v2"

    for dataset in datasets:
        print(f"\n{'='*50}")
        calculate_similarity_for_dataset(dataset, model_name=model_name)
        print(f"{'='*50}")

    print("\n所有数据集处理完成!")
    print("结果文件保存在 node_similarity/ 文件夹下")

if __name__ == "__main__":
    main()
