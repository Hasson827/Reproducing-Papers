import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np
from models import GCN, GAT, GraphSAGE

# 设置随机种子以便复现结果
torch.manual_seed(42)

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 模型设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(
    in_channels=dataset.num_features, 
    hidden_channels=64, 
    out_channels=dataset.num_classes
).to(device)
# 也可以尝试其他模型:
# model = GAT(in_channels=dataset.num_features, hidden_channels=8, out_channels=dataset.num_classes)
# model = GraphSAGE(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
    return acc

# 训练和评估模型
train_losses = []
val_accs = []
test_accs = []

for epoch in range(1, 201):
    loss = train()
    train_losses.append(loss)
    val_acc = evaluate(data.val_mask)
    test_acc = evaluate(data.test_mask)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accs, label='Validation Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()

# 输出最终结果
print(f"Final Test Accuracy: {test_accs[-1]:.4f}")

# 可视化节点嵌入
@torch.no_grad()
def visualize_embeddings():
    from sklearn.manifold import TSNE
    model.eval()
    out = model(data.x, data.edge_index)
    z = out.cpu().numpy()
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=data.y.cpu().numpy(), cmap='tab10', s=70)
    plt.colorbar()
    plt.title('Node Embeddings')
    plt.savefig('node_embeddings.png')
    plt.show()

visualize_embeddings()
