import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class ImprovedGraphSAGE(nn.Module):
    """
    改进的GraphSAGE模型，带有批归一化和跳跃连接
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3, aggregator=None):
        """
        初始化GraphSAGE模型
        
        参数:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            dropout: Dropout概率
            aggregator: 聚合器函数(可选)
        """
        super(ImprovedGraphSAGE, self).__init__()
        
        # 输入变换
        self.input_transform = nn.Linear(in_dim, hidden_dim)
        
        # MLP层作为备选路径
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # GraphSAGE卷积层
        self.conv1 = SAGEConv(hidden_dim, hidden_dim, normalize=True, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, out_dim, normalize=True, aggr='mean')
        
        # 批归一化层增强数值稳定性
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
        # 跳跃连接需要的变换
        self.skip_transform = nn.Linear(hidden_dim, out_dim)
        
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)
        
        # 自定义聚合器
        self.aggregator = aggregator
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')
                else:
                    nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播，带有额外的错误处理和MLP备选路径
        
        参数:
            x: 节点特征
            edge_index: 边索引
            edge_weight: 边权重(可选)
            
        返回:
            节点嵌入
        """
        # 检查NaN值
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # 打印调试信息
        print(f"输入形状: x={x.shape}, edge_index={edge_index.shape}")
        
        # 应用特征变换
        x = self.input_transform(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # 应用批归一化
        x = self.batch_norm1(x)
        
        # 存储用于跳跃连接
        x_skip = x
        
        # 尝试使用GNN层
        use_gnn = True
        
        try:
            # 第一个GraphSAGE层
            if edge_weight is not None:
                x1 = self.conv1(x, edge_index, edge_weight)
            else:
                x1 = self.conv1(x, edge_index)
            
            x1 = F.leaky_relu(x1, negative_slope=0.2)
            x1 = self.dropout(x1)
            
            # 第二个GraphSAGE层
            if edge_weight is not None:
                x2 = self.conv2(x1, edge_index, edge_weight)
            else:
                x2 = self.conv2(x1, edge_index)
            
            # GNN成功
            print("GraphSAGE层成功应用")
            x_gnn = x2
            
        except Exception as e:
            print(f"GNN层出错: {e}")
            use_gnn = False
        
        # 如果GNN失败，使用MLP备选路径
        if not use_gnn:
            print("使用MLP备选路径")
            h = self.hidden_layer(x_skip)
            h = F.leaky_relu(h, negative_slope=0.2)
            h = self.dropout(h)
            x_gnn = self.output_layer(h)
            
        # 跳跃连接的变换
        x_skip_transformed = self.skip_transform(x_skip)
        
        # 确保维度匹配后再相加
        if x_gnn.shape[1] == x_skip_transformed.shape[1]:
            print(f"添加跳跃连接: {x_gnn.shape} + {x_skip_transformed.shape}")
            x = x_gnn + x_skip_transformed
        else:
            print(f"维度不匹配，无法添加跳跃连接: {x_gnn.shape} vs {x_skip_transformed.shape}")
            # 使用正确维度的张量
            x = x_gnn
        
        # 最终批归一化
        try:
            x = self.batch_norm2(x)
        except Exception as e:
            print(f"批归一化出错: {e}")
            # 跳过批归一化
        
        # 替换可能出现的NaN值
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # L2归一化嵌入
        x = F.normalize(x, p=2, dim=1)
        
        return x
