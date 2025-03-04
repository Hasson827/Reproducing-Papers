import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 注意力头的数量
        self.d_model = d_model      # 模型的维度
        
        # 确保模型维度可以被头数整除
        assert d_model % num_heads == 0
        
        # 每个头的维度
        self.depth = d_model // num_heads
        
        # 线性映射层
        self.wq = nn.Linear(d_model, d_model)  # 查询映射
        self.wk = nn.Linear(d_model, d_model)  # 键映射
        self.wv = nn.Linear(d_model, d_model)  # 值映射
        
        # 输出映射
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        """
        将输入张量分割成多个头
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            batch_size: 批次大小
            
        返回:
            分割后的张量 [batch_size, num_heads, seq_len, depth]
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, v, k, q, mask=None):
        """
        前向传播
        
        参数:
            v: 值 [batch_size, seq_len, d_model]
            k: 键 [batch_size, seq_len, d_model]
            q: 查询 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
            
        返回:
            注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        
        # 线性映射
        q = self.wq(q)  # [batch_size, seq_len, d_model]
        k = self.wk(k)  # [batch_size, seq_len, d_model]
        v = self.wv(v)  # [batch_size, seq_len, d_model]
        
        # 分割多头
        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)  # [batch_size, num_heads, seq_len_k, depth]
        v = self.split_heads(v, batch_size)  # [batch_size, num_heads, seq_len_v, depth]
        
        # 计算注意力权重
        # 缩放点积注意力
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 缩放
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        
        # 掩码处理
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        # softmax归一化
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 加权求和
        output = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len_q, depth]
        
        # 重塑张量
        output = output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len_q, num_heads, depth]
        output = output.view(batch_size, -1, self.d_model)  # [batch_size, seq_len_q, d_model]
        
        # 输出映射
        output = self.dense(output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """
    前馈神经网络实现
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层
        self.relu = nn.ReLU()                    # ReLU激活函数
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 [batch_size, seq_len, d_model]
            
        返回:
            输出 [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)  # [batch_size, seq_len, d_ff]
        x = self.relu(x)     # [batch_size, seq_len, d_ff]
        x = self.linear2(x)  # [batch_size, seq_len, d_model]
        return x

def create_padding_mask(seq):
    """
    创建填充掩码，用于掩盖序列中的填充标记
    
    参数:
        seq: 输入序列 [batch_size, seq_len]
        
    返回:
        掩码 [batch_size, 1, 1, seq_len]
    """
    # 将0(PAD)位置标记为1，其他位置为0
    mask = (seq == 0).float().unsqueeze(1).unsqueeze(2)
    return mask  # [batch_size, 1, 1, seq_len]

def create_look_ahead_mask(size):
    """
    创建前瞻掩码，用于掩盖序列中的未来位置
    
    参数:
        size: 序列长度
        
    返回:
        掩码 [size, size]
    """
    # 创建上三角矩阵 (对角线为0)
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask  # [size, size]

class DecoderLayer(nn.Module):
    """
    Transformer解码器层实现
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        # 掩码多头自注意力
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        
        # 编码器-解码器多头注意力
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        # 前馈神经网络
        self.ffn = FeedForward(d_model, d_ff)
        
        # 层归一化
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        """
        前向传播
        
        参数:
            x: 解码器输入 [batch_size, target_seq_len, d_model]
            enc_output: 编码器输出 [batch_size, input_seq_len, d_model]
            look_ahead_mask: 前瞻掩码，防止关注未来位置
            padding_mask: 填充掩码，防止关注填充位置
            
        返回:
            解码器层输出 [batch_size, target_seq_len, d_model]
        """
        # 第一个多头自注意力（掩码）
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)  # (batch_size, target_seq_len, d_model)
        
        # 第二个多头注意力（编码器-解码器）
        # 注意：这里的查询来自解码器第一层，键和值来自编码器输出
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)  # (batch_size, target_seq_len, d_model)
        
        # 前馈神经网络
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)  # (batch_size, target_seq_len, d_model)
        
        return out3, attn_weights_block1, attn_weights_block2

class Decoder(nn.Module):
    """
    Transformer解码器实现（由多个解码器层堆叠而成）
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, 
                 maximum_position_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 词嵌入
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)
        
        # 解码器层
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        """
        前向传播
        
        参数:
            x: 目标序列 [batch_size, target_seq_len]
            enc_output: 编码器输出 [batch_size, input_seq_len, d_model]
            look_ahead_mask: 前瞻掩码
            padding_mask: 填充掩码
            
        返回:
            解码器输出 [batch_size, target_seq_len, d_model]
            注意力权重
        """
        seq_len = x.size(1)
        attention_weights = {}
        
        # 词嵌入
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        
        # 缩放嵌入
        x = x * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 解码器层
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        # x.shape: (batch_size, target_seq_len, d_model)
        return x, attention_weights

class PositionalEncoding(nn.Module):
    """
    位置编码实现
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 [batch_size, seq_len, d_model]
            
        返回:
            位置编码后的输入 [batch_size, seq_len, d_model]
        """
        # 添加位置编码到输入
        x = x + self.pe[:x.size(0), :]
        return x

# 示例使用
def example_usage():
    """
    展示解码器的使用示例
    """
    # 参数设置
    batch_size = 64
    input_seq_len = 30
    target_seq_len = 20
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    target_vocab_size = 10000
    dropout_rate = 0.1
    
    # 创建示例输入
    x = torch.randint(1, target_vocab_size, (batch_size, target_seq_len))  # 目标序列
    enc_output = torch.randn((batch_size, input_seq_len, d_model))  # 编码器输出
    
    # 创建掩码
    look_ahead_mask = create_look_ahead_mask(target_seq_len).to(x.device)
    padding_mask = create_padding_mask(torch.zeros((batch_size, input_seq_len))).to(x.device)
    
    # 创建解码器
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        target_vocab_size=target_vocab_size,
        maximum_position_encoding=5000,
        dropout_rate=dropout_rate
    )
    
    # 前向传播
    output, attention_weights = decoder(x, enc_output, look_ahead_mask, padding_mask)
    
    print(f"解码器输出形状: {output.shape}")  # [batch_size, target_seq_len, d_model]
    print(f"注意力权重字典大小: {len(attention_weights)}")  # 2 * num_layers
    
    # 查看部分注意力权重形状
    for name, weight in list(attention_weights.items())[:2]:
        print(f"{name} 形状: {weight.shape}")

if __name__ == "__main__":
    example_usage()
