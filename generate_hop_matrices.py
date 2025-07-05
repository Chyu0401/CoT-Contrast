import torch
import numpy as np
import networkx as nx
import os
from typing import Dict, List, Tuple

def load_anchors(anchors_path: str) -> List[int]:
    anchors = torch.load(anchors_path)
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.tolist()
    return anchors

def load_graph_data(data_path: str) -> Tuple[nx.Graph, int]:
    data = torch.load(data_path)
    
    edge_index = data['edge_index']
    num_nodes = edge_index.max().item() + 1
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    
    print(f"成功加载图数据: {data_path}")
    print(f"   - 节点数: {num_nodes}")
    print(f"   - 边数: {G.number_of_edges()}")
    
    return G, num_nodes

def bfs_hop_matrix(G: nx.Graph, anchor_node: int, max_hop: int, num_nodes: int) -> np.ndarray:
    hop_matrix = np.zeros((max_hop, num_nodes), dtype=np.int32)
    
    shortest_paths = nx.single_source_shortest_path_length(G, anchor_node, cutoff=max_hop)
    
    for node, distance in shortest_paths.items():
        if distance > 0 and distance <= max_hop:  # 距离为0表示锚点本身，不记录
            hop_matrix[distance - 1][node] = 1  # distance-1 因为数组索引从0开始
    
    return hop_matrix

def compute_all_hop_matrices(G: nx.Graph, anchors: List[int], max_hop: int, num_nodes: int) -> Dict[int, np.ndarray]:
    all_hop_matrices = {}
    
    print(f"开始计算 {len(anchors)} 个锚点的跳数矩阵...")
    
    for i, anchor in enumerate(anchors):
        if i % 10 == 0:  # 每10个锚点打印一次进度
            print(f"  处理进度: {i}/{len(anchors)}")
            
        hop_matrix = bfs_hop_matrix(G, anchor, max_hop, num_nodes)
        all_hop_matrices[anchor] = hop_matrix
    
    print(f"完成所有锚点的跳数矩阵计算!")
    return all_hop_matrices

def save_hop_matrices(hop_matrices: Dict[int, np.ndarray], output_path: str):
    torch.save(hop_matrices, output_path)
    print(f"跳数矩阵已保存到: {output_path}")

def main():

    dataset_configs = {
        'cora': {
            'max_hop': 4,
            'anchors_path': 'anchors/cora_anchors.pt',
            'data_path': 'datasets/cora.pt',
            'output_path': 'hop_matrices/hop_matrices_cora.pt'
        },
        'citeseer': {
            'max_hop': 5,
            'anchors_path': 'anchors/citeseer_anchors.pt',
            'data_path': 'datasets/citeseer.pt',
            'output_path': 'hop_matrices/hop_matrices_citeseer.pt'
        }
    }
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n{'='*50}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*50}")
        
      
        anchors_path = config['anchors_path']
        data_path = config['data_path']
        output_path = config['output_path']
        max_hop = config['max_hop']
        
        
        if not os.path.exists(anchors_path):
            print(f"警告: 锚点文件 {anchors_path} 不存在，跳过此数据集")
            continue
            
        if not os.path.exists(data_path):
            print(f"警告: 图数据文件 {data_path} 不存在，跳过此数据集")
            continue
        
        try:
           
            print("步骤1: 读取锚点...")
            anchors = load_anchors(anchors_path)
            print(f"  加载了 {len(anchors)} 个锚点")
            
           
            print("步骤2: 加载图数据...")
            G, num_nodes = load_graph_data(data_path)
            
            
            print("步骤3: 计算跳数矩阵...")
            hop_matrices = compute_all_hop_matrices(G, anchors, max_hop, num_nodes)
            
            
            print("步骤4: 保存结果...")
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_hop_matrices(hop_matrices, output_path)
            
          
            print(f"\n统计信息:")
            print(f"   - 锚点数量: {len(anchors)}")
            print(f"   - 最大跳数: {max_hop}")
            print(f"   - 节点数量: {num_nodes}")
            print(f"   - 跳数矩阵形状: {max_hop} x {num_nodes}")
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
            continue

if __name__ == "__main__":
    main()
