import torch
import pickle
import networkx as nx
import os
import time
from collections import defaultdict

def load_data(dataset_name):
    print(f"加载数据集: datasets/{dataset_name}.pt")
    data = torch.load(f'datasets/{dataset_name}.pt', weights_only=False)
    print("成功加载数据集")
    print("数据集信息:")
    print(f"   - 数据类型: {type(data)}")
    print(f"   - 字典键: {list(data.keys())}")
    
    edge_index = data['edge_index']
    num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1
    num_edges = edge_index.shape[1]
    num_labels = len(data['y']) if 'y' in data else 0
    
    print(f"   - 推断节点数: {num_nodes}")
    print(f"   - 边数: {num_edges}")
    print(f"   - 标签数量: {num_labels}")
    
    return data

def create_networkx_graph(data):
    print("转换为NetworkX图...")
    
    edge_index = data['edge_index']
    G = nx.Graph()
    
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        G.add_edge(src, dst)
    
    print(f"NetworkX图创建完成:")
    print(f"   - 节点数: {G.number_of_nodes()}")
    print(f"   - 边数: {G.number_of_edges()}")
    print(f"   - 是否连通: {nx.is_connected(G)}")
    
    return G

def load_anchors_and_hop_matrices(dataset_name):
    print(f"加载锚点和跳数矩阵...")
    
    anchors_path = f'anchors/{dataset_name}_anchors.pt'
    if os.path.exists(anchors_path):
        anchors = torch.load(anchors_path, weights_only=False).tolist()
        print(f"   - 锚点数量: {len(anchors)}")
        print(f"   - 锚点列表: {anchors[:10]}{'...' if len(anchors) > 10 else ''}")
    else:
        raise FileNotFoundError(f"锚点文件不存在: {anchors_path}")
    
    hop_matrices_path = f'hop_matrices/hop_matrices_{dataset_name}.pt'
    if os.path.exists(hop_matrices_path):
        anchor_hop_matrices = torch.load(hop_matrices_path, weights_only=False)
        print(f"   - 跳数矩阵数量: {len(anchor_hop_matrices)}")
        max_hop = len(next(iter(anchor_hop_matrices.values())))
        print(f"   - 最大跳数: {max_hop}")
    else:
        raise FileNotFoundError(f"跳数矩阵文件不存在: {hop_matrices_path}")
    
    return anchors, anchor_hop_matrices, max_hop

def calculate_shortest_paths(G, anchors, anchor_hop_matrices, max_hop):
    print("开始计算最短路径...")
    
    all_paths = {}
    total_paths = 0
    
    for i, anchor in enumerate(anchors):
        print(f"处理锚点 {i+1}/{len(anchors)}: {anchor}")
        anchor_paths = []
        
        for d in range(max_hop):
            hop_vec = anchor_hop_matrices[anchor][d]
            target_nodes = torch.where(torch.tensor(hop_vec) == 1)[0].tolist()
            
            for target in target_nodes:
                if target == anchor:
                    continue
                    
                try:
                    path = nx.shortest_path(G, source=anchor, target=target)
                    anchor_paths.append((target, path))
                    total_paths += 1
                except nx.NetworkXNoPath:
                    print(f"    警告: 锚点 {anchor} 到节点 {target} 无路径")
                    continue
        
        all_paths[anchor] = anchor_paths
        print(f"  锚点 {anchor} 完成，找到 {len(anchor_paths)} 条路径")
    
    print(f"最短路径计算完成，总共找到 {total_paths} 条路径")
    return all_paths

def save_paths(all_paths, dataset_name):
    print("保存路径结果...")
    
    os.makedirs('shortest_paths', exist_ok=True)
    
    torch_path = f'shortest_paths/{dataset_name}_shortest_paths.pt'
    torch.save(all_paths, torch_path)
    print(f"   - 保存为torch格式: {torch_path}")
    
    return torch_path

def analyze_paths(all_paths):
    print("分析路径统计信息...")
    
    total_anchors = len(all_paths)
    total_paths = sum(len(paths) for paths in all_paths.values())
    
    path_lengths = []
    for anchor_paths in all_paths.values():
        for target, path in anchor_paths:
            path_lengths.append(len(path) - 1)
    
    if path_lengths:
        avg_length = sum(path_lengths) / len(path_lengths)
        min_length = min(path_lengths)
        max_length = max(path_lengths)
        
        print(f"   - 锚点数量: {total_anchors}")
        print(f"   - 总路径数: {total_paths}")
        print(f"   - 平均路径长度: {avg_length:.2f}")
        print(f"   - 最短路径长度: {min_length}")
        print(f"   - 最长路径长度: {max_length}")
        
        length_dist = defaultdict(int)
        for length in path_lengths:
            length_dist[length] += 1
        
        print("   - 路径长度分布:")
        for length in sorted(length_dist.keys()):
            print(f"     {length}跳: {length_dist[length]}条路径")
    else:
        print("   - 没有找到任何路径")

def show_examples(all_paths, dataset_name):
    """显示路径矩阵的示例格式"""
    print(f"\n{dataset_name} 数据集 - 路径矩阵示例:")
    print("=" * 60)
    
    if not all_paths:
        print("没有找到任何路径")
        return
    
    # 显示前3个锚点的示例
    anchor_count = 0
    for anchor, anchor_paths in all_paths.items():
        if anchor_count >= 3:  
            break
            
        print(f"\n锚点 {anchor} 的路径:")
        print(f"   找到 {len(anchor_paths)} 条路径")
        
        # 显示前5条路径作为示例
        for i, (target, path) in enumerate(anchor_paths[:5]):
            path_length = len(path) - 1
            print(f"   {i+1:2d}. 锚点{anchor} → 目标{target:3d}: {path} (长度: {path_length}跳)")
        
        if len(anchor_paths) > 5:
            print(f"   ... 还有 {len(anchor_paths) - 5} 条路径")
        
        anchor_count += 1
    
    print(f"\n数据结构说明:")
    print(f"   all_paths = {{")
    print(f"       anchor_id: [")
    print(f"           (target_node, [anchor, ..., target]),")
    print(f"           ...")
    print(f"       ],")
    print(f"       ...")
    print(f"   }}")
    

    print(f"\n具体示例:")
    first_anchor = list(all_paths.keys())[0]
    if all_paths[first_anchor]:
        first_path = all_paths[first_anchor][0]
        target, path = first_path
        print(f"   all_paths[{first_anchor}] = [")
        print(f"       ({target}, {path}),")
        print(f"       ...")
        print(f"   ]")
    
    print("=" * 60)

def main():
    
    datasets = ['cora', 'citeseer']
    
    for dataset_name in datasets:
        print(f"\n处理数据集: {dataset_name}")
        print("-" * 30)
        
        try:
            data = load_data(dataset_name)
            G = create_networkx_graph(data)
            anchors, anchor_hop_matrices, max_hop = load_anchors_and_hop_matrices(dataset_name)
            
            start_time = time.time()
            all_paths = calculate_shortest_paths(G, anchors, anchor_hop_matrices, max_hop)
            end_time = time.time()
            
            print(f"计算耗时: {end_time - start_time:.2f}秒")
            
            torch_path = save_paths(all_paths, dataset_name)
            analyze_paths(all_paths)
            # show_examples(all_paths, dataset_name)
            
            print(f"数据集 {dataset_name} 处理完成!")
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("第三阶段完成!")
    print("=" * 50)

if __name__ == "__main__":
    main() 