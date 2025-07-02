import torch
import json
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import os

def anchor_selection(data_dict, c=2, CR=0.6):

    # 从字典中提取图数据
    edge_index = data_dict['edge_index']
    
    # 从edge_index推断节点数量
    if 'num_nodes' in data_dict:
        num_nodes = data_dict['num_nodes']
    else:
        # 从edge_index中获取最大节点ID + 1
        num_nodes = edge_index.max().item() + 1
    
    V = set(range(num_nodes))
    A = set()       # 锚点集合
    N_cover = set() # 被覆盖的节点集合

    # 预先缓存每个节点的 c-hop 邻居
    print("预计算每个节点的 c-hop 邻居...")
    c_hop_neighbors = {}
    for v in tqdm(V):
        subset, _, _, _ = k_hop_subgraph(v, c, edge_index, relabel_nodes=False)
        c_hop_neighbors[v] = set(subset.tolist())

    # 贪心选择锚点
    print("开始选择锚点...")
    target_cover = int(CR * num_nodes)
    
    while len(N_cover) < target_cover:
        max_gain = -1
        best_anchor = None

        for v in V - A:
            new_covered = c_hop_neighbors[v] - N_cover
            if len(new_covered) > max_gain:
                max_gain = len(new_covered)
                best_anchor = v

        if max_gain == 0 or best_anchor is None:
            print("无更多可覆盖节点，提前结束")
            break

        A.add(best_anchor)
        N_cover.update(c_hop_neighbors[best_anchor])
        
        # 打印进度
        current_cover_rate = len(N_cover) / num_nodes
        print(f"当前覆盖率: {current_cover_rate:.3f} ({len(N_cover)}/{num_nodes})")

    return list(A)

def save_anchors(anchors, dataset_name, output_dir="anchors"):


    os.makedirs(output_dir, exist_ok=True)
    

    pt_path = os.path.join(output_dir, f"{dataset_name}_anchors.pt")
    anchors_tensor = torch.tensor(anchors, dtype=torch.long)
    torch.save(anchors_tensor, pt_path)
    print(f"锚点已保存到: {pt_path}")
    

    json_path = os.path.join(output_dir, f"{dataset_name}_anchors.json")
    with open(json_path, "w") as f:
        json.dump(anchors, f)
    print(f"锚点也保存为JSON格式: {json_path}")
    
    print(f"锚点数量: {len(anchors)}")

def process_dataset(dataset_path, dataset_name, c, CR):

    print(f"\n{'='*50}")
    print(f"处理数据集: {dataset_name}")
    print(f"参数设置: c={c}, CR={CR}")
    print(f"{'='*50}")
    

    if not os.path.exists(dataset_path):
        print(f"错误：找不到文件 {dataset_path}")
        return None
    

    print(f"加载数据集: {dataset_path}")
    data_dict = torch.load(dataset_path)
    print(f"成功加载数据集")
    

    print(f"数据集信息:")
    print(f"   - 数据类型: {type(data_dict)}")
    if isinstance(data_dict, dict):
        print(f"   - 字典键: {list(data_dict.keys())}")
        if 'num_nodes' in data_dict:
            print(f"   - 节点数: {data_dict['num_nodes']}")
        else:
            # 从edge_index推断节点数
            num_nodes = data_dict['edge_index'].max().item() + 1
            print(f"   - 推断节点数: {num_nodes}")
        if 'edge_index' in data_dict:
            print(f"   - 边数: {data_dict['edge_index'].shape[1]}")
        if 'x' in data_dict:
            print(f"   - 特征维度: {data_dict['x'].shape[1]}")
        if 'y' in data_dict:
            print(f"   - 标签数量: {len(data_dict['y'])}")
    else:
        print(f"   - 节点数: {data_dict.num_nodes}")
        print(f"   - 边数: {data_dict.edge_index.shape[1]}")
        print(f"   - 特征维度: {data_dict.x.shape[1]}")
    
    # 选择锚点
    print(f"\n开始锚点选择...")
    anchors = anchor_selection(data_dict, c=c, CR=CR)
    
    # 保存锚点
    print(f"\n保存锚点结果...")
    save_anchors(anchors, dataset_name)
    
    # 打印统计信息
    print(f"\n锚点选择完成!")
    print(f"   - 选出的锚点数量: {len(anchors)}")
    
    # 计算锚点比例
    if isinstance(data_dict, dict):
        if 'num_nodes' in data_dict:
            num_nodes = data_dict['num_nodes']
        else:
            num_nodes = data_dict['edge_index'].max().item() + 1
    else:
        num_nodes = data_dict.num_nodes
    
    print(f"   - 锚点比例: {len(anchors)/num_nodes:.3f}")
    print(f"   - 示例锚点ID: {anchors[:10]}")
    
    return anchors

def main():

    datasets_config = [
        {
            "path": "datasets/cora.pt",
            "name": "cora",
            "c": 2,      
            "CR": 0.6    
        },
        {
            "path": "datasets/citeseer.pt", 
            "name": "citeseer",
            "c": 2,      
            "CR": 0.5    
        }
    ]
    
    print("开始处理两个数据集的锚点选择...")
    print("各数据集参数设置:")
    for config in datasets_config:
        print(f"  - {config['name']}: c={config['c']}, CR={config['CR']}")
    
    # 处理每个数据集
    results = {}
    for config in datasets_config:
        anchors = process_dataset(
            dataset_path=config["path"],
            dataset_name=config["name"],
            c=config["c"],
            CR=config["CR"]
        )
        if anchors is not None:
            results[config["name"]] = anchors
    

    print(f"\n{'='*60}")
    print(f"所有数据集处理完成!")
    print(f"{'='*60}")
    for dataset_name, anchors in results.items():
        print(f"{dataset_name}: {len(anchors)} 个锚点")
    
    print(f"\n所有锚点文件已保存到 'anchors/' 目录")
    print(f"   - cora_anchors.pt / cora_anchors.json")
    print(f"   - citeseer_anchors.pt / citeseer_anchors.json")

if __name__ == "__main__":
    main()
