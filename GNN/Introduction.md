# 图神经网络(GNN)从入门到实践

## 1. 图论基础

### 1.1 什么是图(Graph)?

图是一种数学结构，用于表示某些事物之间的"关系"。一个图 G 由两个集合组成：

- **节点集合 V**（也称为顶点）
- **边集合 E**（连接节点的线）

形式化表示为：G = (V, E)

### 1.2 图的类型

- **无向图**: 边没有方向
- **有向图**: 边有特定方向
- **加权图**: 边具有权重值
- **二部图**: 节点可分为两个独立集合
- **多重图**: 节点之间可以有多条边
- **超图**: 一条边可以连接多个节点

### 1.3 图的表示方法

#### 邻接矩阵 (Adjacency Matrix)

对于有n个节点的图，邻接矩阵是一个n×n的矩阵A，其中：

- A[i][j] = 1 表示节点i和节点j之间有边
- A[i][j] = 0 表示节点i和节点j之间没有边
- 对于加权图，A[i][j]可以是边的权重

示例：

```
A = [
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
]
```

#### 邻接表 (Adjacency List)

每个节点都维护一个列表，列表中包含与其相连的所有节点。

示例：

```
节点0: [1, 2]
节点1: [0, 2]
节点2: [0, 1, 3]
节点3: [2]
```

#### 边列表 (Edge List)

简单地存储所有边的列表。

示例：

```
边: [(0,1), (0,2), (1,2), (2,3)]
```

## 2. 从传统图算法到图神经网络

### 2.1 为什么需要图神经网络?

传统图算法（如最短路径、社区发现等）在处理以下情况时遇到挑战：

1. **复杂特征**: 现实世界的节点和边通常具有丰富的特征信息
2. **高维模式识别**: 识别图中的复杂模式和非线性关系
3. **归纳学习**: 将学到的知识应用到未见过的图结构
4. **端到端学习**: 直接从原始数据学习任务相关的表示

图神经网络(GNN)能够应对这些挑战，因为它们：

- 能处理节点和边的丰富特征
- 可以学习图结构中复杂的非线性模式
- 具有归纳能力（应用到新节点/新图）
- 支持端到端的学习

### 2.2 图上的机器学习任务

使用GNN可以解决的典型任务：

1. **节点级任务**:
   - 节点分类（如社交网络中的用户分类）
   - 节点回归（如预测交通网络中的流量）

2. **边级任务**:
   - 链接预测（预测两个节点之间是否存在连接）
   - 边分类（对关系类型进行分类）

3. **图级任务**:
   - 图分类（如分子分类，判断药物毒性）
   - 图回归（如预测分子性质）
   - 图生成（生成新的有效分子结构）

## 3. 图神经网络的基本原理

### 3.1 核心思想：消息传递

GNN的核心是**消息传递**机制，可以概括为以下步骤：

1. **收集邻居信息**: 每个节点从其邻居收集信息
2. **信息聚合**: 将收集到的信息进行聚合
3. **节点状态更新**: 基于聚合的信息更新节点自身的表示

这个过程可以迭代多次，使得信息能够传播到更远的节点。

### 3.2 图卷积网络(GCN)的数学基础

图卷积网络是最基础的GNN模型之一。对于节点v，其第l+1层的表示可以通过以下公式计算：

$$
h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{d_u d_v}} W^{(l)} h_u^{(l)}\right)
$$

其中:

- $h_v^{(l)}$ 是节点v在第l层的特征表示
- $\mathcal{N}(v)$ 是节点v的邻居集合
- $d_v$ 和 $d_u$ 分别是节点v和u的度
- $W^{(l)}$ 是第l层的可学习权重矩阵
- $\sigma$ 是非线性激活函数(如ReLU)

简化的矩阵形式:

$$
H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})
$$

其中:

- $\tilde{A} = A + I_N$ 是添加了自环的邻接矩阵
- $\tilde{D}$ 是 $\tilde{A}$ 的度矩阵
- $H^{(l)}$ 是所有节点在第l层的特征矩阵

### 3.3 其他GNN变体

#### 图注意力网络 (GAT)

GAT通过注意力机制为不同邻居分配不同权重:

$$
h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W^{(l)} h_u^{(l)}\right)
$$

其中 $\alpha_{vu}$ 是通过注意力计算的权重系数。

#### GraphSAGE

GraphSAGE使用采样和聚合来处理大规模图:

$$
h_v^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{AGGREGATE}(\{h_v^{(l)}\} \cup \{h_u^{(l)}, \forall u \in \mathcal{N}(v)\})\right)
$$

其中AGGREGATE可以是各种聚合函数，如均值、最大值或LSTM。

## 4. GNN的PyTorch代码实现

### 4.1 GCN实现

以下是使用PyTorch Geometric实现GCN的核心代码:

```python
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
      
        # 第二层GCN
        x = self.conv2(x, edge_index)
      
        return x
```

GCNConv的实现原理:

```python
# PyTorch Geometric中GCNConv的简化版实现
def gcn_conv(x, edge_index, edge_weight=None):
    # 第1步：添加自环
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, num_nodes=x.size(0))
  
    # 第2步：计算归一化系数
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
  
    # 第3步：消息传递和聚合
    return scatter_add(norm.view(-1, 1) * x[col], row, dim=0, dim_size=x.size(0))
```

### 4.2 GAT实现

以下是实现图注意力网络(GAT)的核心代码:

```python
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        # 最后一层使用单头注意力
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
      
        return x
```

注意力权重的计算方式:

```python
# 计算注意力权重的简化版实现
def compute_attention(query, key, edge_index):
    # query/key: [num_nodes, channels]
    # 计算注意力分数
    row, col = edge_index
    alpha = (query[row] * key[col]).sum(dim=-1)
  
    # 对每个节点的邻居进行softmax归一化
    alpha = softmax(alpha, row, num_nodes=query.size(0))
  
    return alpha
```

### 4.3 GraphSAGE实现

GraphSAGE的核心实现:

```python
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
      
        return x
```

SAGEConv的聚合过程:

```python
# SAGEConv的简化版实现
def sage_conv(x, edge_index):
    # 收集邻居信息
    row, col = edge_index
  
    # 聚合邻居节点特征 (这里使用mean aggregation)
    out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
  
    # 与自身特征连接并线性变换
    out = torch.cat([x, out], dim=1)
  
    return out
```

## 5. 训练GNN模型

### 5.1 节点分类任务训练流程

```python
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
  
    # 前向传播
    out = model(data.x, data.edge_index)
  
    # 计算损失 (仅使用训练节点)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
  
    # 反向传播和优化
    loss.backward()
    optimizer.step()
  
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
    return acc
```

### 5.2 实际训练过程

完整的训练循环示例:

```python
# 模型和优化器设置
model = GCN(in_channels=dataset.num_features, 
            hidden_channels=64, 
            out_channels=dataset.num_classes)
          
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练循环
for epoch in range(1, 201):
    loss = train(model, data, optimizer)
    val_acc = evaluate(model, data, data.val_mask)
    test_acc = evaluate(model, data, data.test_mask)
  
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
```

## 6. GNN的应用案例

### 6.1 引文网络分类 (Cora数据集)

Cora是一个包含科学出版物的引文网络:

- 节点: 科学论文
- 边: 引用关系
- 节点特征: 论文中单词的二元表示
- 任务: 根据内容和引用关系对论文进行分类

使用GNN处理Cora数据集:

```python
# 加载数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 模型训练和评估
# ...见前面的训练代码...
```

### 6.2 自定义图数据处理

创建和处理自定义图数据:

```python
# 创建自定义图
edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6],  # 源节点
    [1, 2, 0, 2, 0, 1, 4, 5, 3, 3, 3]   # 目标节点
], dtype=torch.long)

# 节点特征
x = torch.randn(7, 10)  # 7个节点，每个有10维特征

# 创建PyG数据对象
data = Data(x=x, edge_index=edge_index)

# 使用模型
model = GCN(in_channels=10, hidden_channels=16, out_channels=2)
output = model(data.x, data.edge_index)
```

## 7. GNN进阶话题

### 7.1 超参数调优

GNN模型常见的超参数:

- 层数: 通常2-3层最有效
- 隐藏维度大小
- Dropout率
- 学习率
- 权重衰减
- 注意力头数(对于GAT)

### 7.2 过平滑问题

当GNN堆叠多层时，节点表示容易趋于相似，这称为"过平滑"问题。解决方法:

- 残差连接
- 跳跃连接
- PairNorm正则化
- 控制GNN层数

### 7.3 异构图处理

在异构图中，存在多种类型的节点和边。处理方法:

- 关系GCN(RGCN)
- 异构图注意力网络(HAN)
- 元路径采样

## 8. 结论与资源

### 8.1 GNN库推荐

- PyTorch Geometric (PyG)
- Deep Graph Library (DGL)
- StellarGraph
- Graph Nets (TensorFlow)

### 8.2 学习资源

- 《图神经网络的深入理解》 (William L. Hamilton)
- CS224W: Stanford大学图机器学习课程
- PyTorch Geometric文档与教程

### 8.3 总结

图神经网络是处理图结构数据的强大工具:

- 融合了图结构和节点特征信息
- 通过消息传递机制学习节点表示
- 广泛应用于社交网络、分子结构、推荐系统等领域
- 仍在快速发展的研究领域
