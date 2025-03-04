import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        参数:
            d_model: 嵌入向量的维度
            max_len: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 创建位置索引
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母中的指数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用余弦
        
        # 添加批次维度
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入嵌入
        
        参数:
            x: 输入嵌入 [batch_size, seq_len, d_model]
            
        返回:
            位置编码后的嵌入 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

def visualize_positional_encoding(d_model=128, max_len=100):
    """
    可视化位置编码
    
    参数:
        d_model: 嵌入向量的维度
        max_len: 要可视化的最大序列长度
    """
    # 计算位置编码
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    # 绘制热图
    plt.figure(figsize=(12, 8))
    plt.imshow(pe, cmap='viridis', aspect='auto')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Sequence Position')
    plt.title('Positional Encoding Visualization')
    plt.colorbar(label='Encoding Value')
    plt.savefig('positional_encoding_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制前几个维度的波形
    plt.figure(figsize=(15, 10))
    dims_to_plot = [0, 1, 4, 5, 20, 21, d_model//2-2, d_model//2-1]
    for i, dim in enumerate(dims_to_plot):
        plt.subplot(len(dims_to_plot)//2, 2, i+1)
        plt.plot(pe[:, dim])
        plt.title(f'Dimension {dim} {"(sin)" if dim % 2 == 0 else "(cos)"}')
        plt.xlabel('Position')
        plt.ylabel('Encoding Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_dimensions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 降维可视化 (使用PCA)
    pca = PCA(n_components=2)
    pe_2d = pca.fit_transform(pe)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pe_2d[:, 0], pe_2d[:, 1], c=np.arange(max_len), cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(label='Sequence Position')
    plt.title('PCA Reduction of Positional Encodings (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('positional_encoding_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_relative_positions(d_model=128):
    """
    演示位置编码如何捕获相对位置信息
    """
    # 创建位置编码
    pe_module = PositionalEncoding(d_model)
    dummy_input = torch.zeros(1, 100, d_model)
    pe = pe_module(dummy_input)[0].detach().numpy()
    
    # 选择几个位置进行比较
    positions = [10, 20, 30, 40, 50]
    position_pairs = [(10, 15), (15, 20), (50, 55), (55, 60)]
    
    # 计算位置向量之间的点积（相似度）
    similarity_matrix = np.zeros((len(positions), len(positions)))
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            similarity_matrix[i, j] = np.dot(pe[pos1], pe[pos2]) / (np.linalg.norm(pe[pos1]) * np.linalg.norm(pe[pos2]))
    
    # 绘制相似度矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(np.arange(len(positions)), positions)
    plt.yticks(np.arange(len(positions)), positions)
    plt.title('Cosine Similarity Between Different Positional Encodings')
    plt.xlabel('Position')
    plt.ylabel('Position')
    plt.savefig('position_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 验证相对位置的相似性
    print("Relative Position Similarity Comparison:")
    for pos1, pos2 in position_pairs:
        # 计算位置向量之间的余弦相似度
        sim = np.dot(pe[pos1], pe[pos2]) / (np.linalg.norm(pe[pos1]) * np.linalg.norm(pe[pos2]))
        print(f"Similarity between position {pos1} and position {pos2}: {sim:.4f}")
    
    # 可视化相对位置的差异向量
    plt.figure(figsize=(15, 12))
    for i, (pos1, pos2) in enumerate(position_pairs):
        plt.subplot(len(position_pairs), 1, i+1)
        plt.plot(pe[pos1] - pe[pos2])
        plt.title(f'Difference Vector Between Position {pos1} and Position {pos2}')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Difference Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('position_difference_vectors.png', dpi=300, bbox_inches='tight')
    plt.show()

def embedding_with_positional_encoding_demo():
    """
    演示嵌入和位置编码的结合
    """
    # 参数设置
    vocab_size = 10000
    d_model = 128
    seq_len = 20
    batch_size = 2
    
    # 创建随机嵌入
    embedding = nn.Embedding(vocab_size, d_model)
    positional_encoding = PositionalEncoding(d_model)
    
    # 模拟输入序列
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 获取嵌入
    token_embeddings = embedding(input_ids)
    
    # 添加位置编码
    encoded = positional_encoding(token_embeddings)
    
    # 可视化单个嵌入向量（批次0，位置0）
    plt.figure(figsize=(10, 6))
    plt.plot(token_embeddings[0, 0].detach().numpy(), label='Token Embedding Only')
    plt.plot(encoded[0, 0].detach().numpy(), label='Embedding + Positional Encoding')
    plt.plot(encoded[0, 0].detach().numpy() - token_embeddings[0, 0].detach().numpy(), 
             label='Positional Encoding Only', linestyle='--')
    plt.legend()
    plt.title('Combination of Embedding Vector and Positional Encoding (Single Token)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.savefig('embedding_with_positional_encoding.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 可视化编码前后的嵌入矩阵
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.imshow(token_embeddings[0].detach().numpy(), aspect='auto', cmap='viridis')
    plt.title('Original Token Embedding Matrix')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Sequence Position')
    plt.colorbar(label='Embedding Value')
    
    plt.subplot(2, 1, 2)
    plt.imshow(encoded[0].detach().numpy(), aspect='auto', cmap='viridis')
    plt.title('Embedding Matrix with Positional Encoding')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Sequence Position')
    plt.colorbar(label='Embedding Value')
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 使用t-SNE可视化位置编码前后的嵌入向量
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    
    # 将批次和序列维度展平以适应t-SNE
    token_emb_flat = token_embeddings.reshape(-1, d_model).detach().numpy()
    encoded_flat = encoded.reshape(-1, d_model).detach().numpy()
    
    # 应用t-SNE降维
    token_emb_tsne = tsne.fit_transform(token_emb_flat)
    
    # 重新初始化t-SNE以保持结果独立
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    encoded_tsne = tsne.fit_transform(encoded_flat)
    
    # 绘制t-SNE结果
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    for i in range(batch_size):
        plt.scatter(
            token_emb_tsne[i*seq_len:(i+1)*seq_len, 0],
            token_emb_tsne[i*seq_len:(i+1)*seq_len, 1],
            c=np.arange(seq_len),
            cmap='viridis',
            alpha=0.7,
            label=f'Batch {i}'
        )
    plt.colorbar(label='Sequence Position')
    plt.title('t-SNE Visualization of Original Token Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(batch_size):
        plt.scatter(
            encoded_tsne[i*seq_len:(i+1)*seq_len, 0],
            encoded_tsne[i*seq_len:(i+1)*seq_len, 1],
            c=np.arange(seq_len),
            cmap='viridis',
            alpha=0.7,
            label=f'Batch {i}'
        )
    plt.colorbar(label='Sequence Position')
    plt.title('t-SNE Visualization of Embeddings with Positional Encoding')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('embedding_tsne_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_frequency_patterns(d_model=128, max_len=100):
    """
    可视化位置编码在不同维度中的频率变化模式
    
    参数:
        d_model: 嵌入向量的维度
        max_len: 序列的最大长度
    """
    # 计算位置编码
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    # 选择一些有代表性的维度来可视化
    dim_to_visualize = [0, 4, 8, 16, 32, 64, d_model-2]
    
    # 绘制不同维度的正弦波
    plt.figure(figsize=(15, 10))
    for i, dim in enumerate(dim_to_visualize):
        plt.subplot(len(dim_to_visualize), 1, i+1)
        if dim % 2 == 0:
            plt.plot(pe[:, dim], label=f'sin, dimension {dim}')
            # 计算理论周期长度
            wavelength = 2 * np.pi / (div_term[dim//2])
            plt.title(f'Sine Wave, Dimension {dim}, Theoretical Period: {wavelength:.1f}')
        else:
            plt.plot(pe[:, dim], label=f'cos, dimension {dim}')
            # 计算理论周期长度
            wavelength = 2 * np.pi / (div_term[(dim-1)//2])
            plt.title(f'Cosine Wave, Dimension {dim}, Theoretical Period: {wavelength:.1f}')
        
        plt.xlabel('Position Index')
        plt.ylabel('Encoding Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('positional_encoding_wavelengths.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 可视化周期长度与维度索引的关系
    wavelengths = []
    dimensions = list(range(0, d_model, 2))  # 只考虑偶数维度
    
    for dim in dimensions:
        wavelength = 2 * np.pi / (div_term[dim//2])
        wavelengths.append(wavelength)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dimensions, wavelengths, marker='o')
    plt.title('Relationship between Wavelength and Dimension in Positional Encoding')
    plt.xlabel('Dimension Index')
    plt.ylabel('Wavelength')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('wavelength_vs_dimension.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """执行所有可视化功能"""
    print("1. Visualizing Positional Encoding Heatmap and Dimension Waveforms")
    visualize_positional_encoding(d_model=128, max_len=100)
    
    print("\n2. Demonstrating How Positional Encoding Captures Relative Position Information")
    demonstrate_relative_positions(d_model=128)
    
    print("\n3. Demonstrating the Combination of Embeddings and Positional Encoding")
    embedding_with_positional_encoding_demo()
    
    print("\n4. Visualizing Frequency Patterns Across Different Dimensions")
    visualize_frequency_patterns(d_model=128, max_len=200)
    
    print("\nPositional encoding visualization complete. All images have been saved.")

if __name__ == "__main__":
    main()