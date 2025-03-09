import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class Aggregation(nn.Module):
    """
    聚合基类
    """
    def __init__(self):
        super(Aggregation, self).__init__()
    
    def forward(self, x, index, ptr=None, dim_size=None, dim=-2):
        """
        前向传播
        
        Args:
            x: 节点特征
            index: 节点索引
            ptr: 指针 (可选)
            dim_size: 维度大小 (可选)
            dim: 聚合维度 (可选)
            
        Returns:
            聚合特征
        """
        return scatter_mean(x, index, dim=dim, dim_size=dim_size)

class LSTMAggregator(Aggregation):
    """
    LSTM聚合器，更精细地处理邻居节点特征
    """
    def __init__(self, hidden_dim):
        """
        初始化LSTM聚合器
        
        Args:
            hidden_dim: LSTM隐藏维度
        """
        super(LSTMAggregator, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 初始化LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 稳定初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data, gain=0.02)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x, index, ptr=None, dim_size=None, dim=-2):
        """
        聚合器前向传播
        
        Args:
            x: 节点特征
            index: 节点索引
            ptr: 指针 (可选)
            dim_size: 维度大小 (可选)
            dim: 聚合维度 (可选)
            
        Returns:
            聚合特征
        """
        # 替换NaN值
        x = torch.nan_to_num(x, nan=0.0)
        
        # 初始化输出张量
        out = torch.zeros((dim_size or index.max().item() + 1, self.hidden_dim), 
                          device=x.device, dtype=x.dtype)
        
        # 分组节点特征
        unique_indices = torch.unique(index)
        for i in unique_indices:
            # 获取当前节点的邻居特征
            mask = (index == i)
            neighbors = x[mask]
            
            # 如果没有邻居，跳过
            if neighbors.size(0) == 0:
                continue
            
            try:
                # 随机打乱邻居，增强鲁棒性
                if neighbors.size(0) > 1:
                    perm = torch.randperm(neighbors.size(0))
                    neighbors = neighbors[perm]
                
                # 调整形状以适应LSTM
                neighbors = neighbors.unsqueeze(0)  # [1, num_neighbors, hidden_dim]
                
                # 用LSTM处理邻居序列 - 保留梯度
                lstm_out, (h_n, _) = self.lstm(neighbors)
                
                # 使用最终隐藏状态作为聚合结果
                out[i] = h_n.squeeze()
                
                # 替换剩余的NaN值
                if torch.isnan(out[i]).any():
                    out[i] = torch.nan_to_num(out[i], nan=0.0)
            except Exception as e:
                # 如果出现任何错误，退回到均值聚合
                out[i] = torch.mean(neighbors, dim=0)
        
        return out

class AttentionAggregator(Aggregation):
    """
    注意力聚合器，为不同邻居分配不同的重要性
    """
    def __init__(self, hidden_dim):
        """
        初始化注意力聚合器
        
        Args:
            hidden_dim: 隐藏维度
        """
        super(AttentionAggregator, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x, index, ptr=None, dim_size=None, dim=-2):
        """
        前向传播
        
        Args:
            x: 节点特征
            index: 节点索引
            ptr: 指针 (可选)
            dim_size: 维度大小 (可选)
            dim: 聚合维度 (可选)
            
        Returns:
            聚合特征
        """
        # 替换NaN值
        x = torch.nan_to_num(x, nan=0.0)
        
        # 初始化输出张量
        out = torch.zeros((dim_size or index.max().item() + 1, self.hidden_dim), 
                          device=x.device, dtype=x.dtype)
        
        # 分组节点特征
        unique_indices = torch.unique(index)
        for i in unique_indices:
            # 获取当前节点的邻居特征
            mask = (index == i)
            neighbors = x[mask]
            
            # 如果没有邻居，跳过
            if neighbors.size(0) == 0:
                continue
            
            try:
                # 计算注意力分数
                attn_scores = self.attention(neighbors)
                attn_weights = F.softmax(attn_scores, dim=0)
                
                # 加权聚合
                weighted_neighbors = neighbors * attn_weights
                out[i] = torch.sum(weighted_neighbors, dim=0)
                
                # 替换任何NaN值
                if torch.isnan(out[i]).any():
                    out[i] = torch.nan_to_num(out[i], nan=0.0)
            except Exception as e:
                # 如有错误，回退到均值聚合
                out[i] = torch.mean(neighbors, dim=0)
        
        return out
