# 新建文件 src/models/gat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ImprovedGAT(nn.Module):
    """使用图注意力网络的靶点优先级排序模型"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, dropout=0.3):
        """
        初始化GAT模型
        
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            heads: 注意力头数量
            dropout: Dropout概率
        """
        super(ImprovedGAT, self).__init__()
        
        # 输入变换
        self.input_transform = nn.Linear(in_dim, hidden_dim)
        
        # 第一个GAT层 - 使用多头注意力
        self.conv1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        
        # 第二个GAT层 - 将多头注意力合并为单一表示
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        
        # 批归一化层增强数值稳定性
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
        # 跳跃连接需要的变换
        self.skip_transform = nn.Linear(hidden_dim, out_dim)
        
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, edge_index):
        """前向传播"""
        # 检查并处理NaN值
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # 特征变换
        x_in = self.input_transform(x)
        x = F.leaky_relu(x_in, negative_slope=0.2)
        
        # 应用批归一化
        x = self.batch_norm1(x)
        
        # 第一个GAT层
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        
        # 保存用于跳跃连接的中间表示
        x_skip = x
        
        # 第二个GAT层
        x = self.conv2(x, edge_index)
        
        # 跳跃连接 - 帮助训练更深层次的网络
        x = x + self.skip_transform(x_skip)
        
        # 批归一化
        x = self.batch_norm2(x)
        
        # 处理可能的NaN值
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # L2归一化嵌入
        x = F.normalize(x, p=2, dim=1)
        
        return x
