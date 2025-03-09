import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from ..training.loss import combined_loss, contrastive_loss

class Trainer:
    """
    模型训练器
    """
    def __init__(self, model, epochs, lr, weight_decay, margin, patience, device=None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            margin: 对比损失间隔
            patience: 早停耐心值
            device: 训练设备
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.margin = margin
        self.patience = patience
        
        # 设置设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)  # 手动实现权重衰减
        
        # 初始化权重
        self.initialize_weights(self.model)
    
    def initialize_weights(self, model):
        """
        初始化模型权重
        
        Args:
            model: 要初始化的模型
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # 只对2+维参数应用Xavier
                    nn.init.xavier_normal_(param.data, gain=0.01)
                else:
                    nn.init.normal_(param.data, std=0.01)  # 对1D权重
            elif 'bias' in name:
                nn.init.zeros_(param.data)  # 将偏置初始化为零
    
    def train(self, data, important_targets):
        """
        训练模型
        
        Args:
            data: 图数据对象
            important_targets: 重要目标字典
            
        Returns:
            训练后的嵌入
        """
        print("训练模型...")
        print(f"使用设备: {self.device}")
        
        # 最佳模型状态和损失跟踪
        best_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        # 将数据移至设备
        data = data.to(self.device)
        
        # 创建正样本对和负样本对
        positive_pairs, negative_pairs = self.create_training_pairs(data, important_targets)
        
        # 训练循环
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            
            # 前向传播
            embeddings = self.model(data.x, data.edge_index, 
                                  data.edge_weight if hasattr(data, 'edge_weight') else None)
            
            # 检查NaN值
            if torch.isnan(embeddings).any():
                print("警告: 嵌入中检测到NaN值。正在尝试恢复...")
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            # 计算组合损失
            loss = combined_loss(
                embeddings, 
                positive_pairs, 
                negative_pairs, 
                data.edge_index,
                data.edge_weight if hasattr(data, 'edge_weight') else None,
                margin=self.margin,
                alpha=0.7,  # 对比损失权重
                beta=0.3    # 结构损失权重
            )
            
            # 手动添加L2正则化
            l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += self.weight_decay * l2_reg
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs}，损失: {loss.item():.4f}")
            
            # 早停检查
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"在第 {epoch} 个epoch提前停止，最佳损失: {best_loss:.4f}")
                    break
        
        # 恢复最佳模型
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        # 评估模式
        self.model.eval()
        
        # 获取最终嵌入
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index,
                                  data.edge_weight if hasattr(data, 'edge_weight') else None)
            
            # 确保没有NaN值
            if torch.isnan(embeddings).any():
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
        
        return embeddings.cpu()
    
    def create_training_pairs(self, data, important_targets):
        """
        创建训练样本对
        
        Args:
            data: 图数据对象
            important_targets: 重要目标字典
            
        Returns:
            正样本对和负样本对
        """
        # 提取边索引
        edge_index = data.edge_index
        
        # 获取所有边作为正样本对
        positive_pairs = edge_index.t()
        
        # 为负样本生成随机对
        num_nodes = data.x.size(0)
        num_neg_samples = positive_pairs.size(0) * 2  # 2倍的负样本
        
        # 随机生成一些负样本对
        neg_src = torch.randint(0, num_nodes, (num_neg_samples,), device=edge_index.device)
        neg_dst = torch.randint(0, num_nodes, (num_neg_samples,), device=edge_index.device)
        negative_pairs = torch.stack([neg_src, neg_dst], dim=1)
        
        # 确保负样本对不在正样本对中
        edge_set = {(int(src), int(dst)) for src, dst in zip(edge_index[0], edge_index[1])}
        valid_neg_pairs = []
        
        for i in range(negative_pairs.size(0)):
            src, dst = int(negative_pairs[i, 0]), int(negative_pairs[i, 1])
            if src != dst and (src, dst) not in edge_set:
                valid_neg_pairs.append(negative_pairs[i])
            
            if len(valid_neg_pairs) >= positive_pairs.size(0):
                break
        
        if valid_neg_pairs:
            negative_pairs = torch.stack(valid_neg_pairs)
        else:
            # 如果没有有效的负样本对，使用随机对（可能包含假负例）
            negative_pairs = torch.stack([
                torch.randperm(num_nodes)[:positive_pairs.size(0)],
                torch.randperm(num_nodes)[:positive_pairs.size(0)]
            ], dim=1).to(edge_index.device)
        
        return positive_pairs, negative_pairs
