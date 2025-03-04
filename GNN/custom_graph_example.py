import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from models import GCN

# 创建一个自定义图
def create_custom_graph():
    # 创建一个小型社交网络图
    # 边列表：每个元素是一个(source, target)元组
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6],  # 源节点
        [1, 2, 0, 2, 0, 1, 4, 5, 3, 3, 3]   # 目标节点
    ], dtype=torch.long)
    
    # 节点特征 (7个节点，每个有10维特征)
    x = torch.randn(7, 10)
    
    # 节点标签 (假设这是一个节点分类任务)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
    
    # 创建PyG图数据对象
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

# 可视化图
def visualize_graph(data):
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    
    edges = data.edge_index.t().tolist()
    for src, dst in edges:
        G.add_edge(src, dst)
    
    plt.figure(figsize=(8, 6))
    # 使用节点标签作为颜色
    colors = ['blue' if label == 0 else 'red' for label in data.y.numpy()]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=colors, 
            node_size=500, font_size=15, font_color='white')
    plt.savefig('custom_graph.png')
    plt.show()
    
# 在自定义图上运行GCN
def run_gcn_on_custom_graph():
    data = create_custom_graph()
    visualize_graph(data)
    
    # 创建一个简单的GCN模型
    model = GCN(in_channels=10, hidden_channels=16, out_channels=2)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
    print("节点预测结果:", pred)
    print("实际节点标签:", data.y)
    
    # 计算准确率
    accuracy = (pred == data.y).sum().item() / data.num_nodes
    print(f"准确率: {accuracy:.4f}")
    
if __name__ == "__main__":
    run_gcn_on_custom_graph()
