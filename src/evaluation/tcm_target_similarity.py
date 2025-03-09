import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DynamicWeightTargetPrioritizer(nn.Module):
    """
    使用动态学习权重的靶点优先级排序
    """
    def __init__(self, embeddings, node_map, reverse_node_map, target_indices, important_targets, 
                init_embedding_weight=0.5, init_importance_weight=0.5):
        """
        初始化目标优先级排序器
        
        Args:
            embeddings: 节点嵌入
            node_map: 节点ID到索引的映射
            reverse_node_map: 索引到节点ID的映射
            target_indices: 靶点索引列表
            important_targets: 重要靶点字典 {target_id: importance_score}
            init_embedding_weight: 嵌入相似度的初始权重
            init_importance_weight: 靶点重要性的初始权重
        """
        super(DynamicWeightTargetPrioritizer, self).__init__()
        
        self.embeddings = embeddings
        self.node_map = node_map
        self.reverse_node_map = reverse_node_map
        self.target_indices = target_indices
        self.important_targets = important_targets
        
        # 初始化可学习的权重参数
        self.embedding_weight = nn.Parameter(torch.tensor(init_embedding_weight))
        self.importance_weight = nn.Parameter(torch.tensor(init_importance_weight))
        
        # 权重应该归一化以和为1
        self._normalize_weights()
        
        # 优化器
        self.optimizer = torch.optim.Adam([self.embedding_weight, self.importance_weight], lr=0.01)
    
    def _normalize_weights(self):
        """
        确保权重和为1
        """
        total = self.embedding_weight + self.importance_weight
        self.embedding_weight.data = self.embedding_weight.data / total
        self.importance_weight.data = self.importance_weight.data / total
    
    def compute_priorities(self, compound_idx, target_indices=None, top_k=None):
        """
        计算给定化合物的靶点优先级
        
        Args:
            compound_idx: 化合物节点索引
            target_indices: 要考虑的靶点索引 (可选)
            top_k: 要返回的顶级靶点数 (可选)
            
        Returns:
            按优先级降序排序的 (target_idx, score) 元组列表
        """
        # 如果没有指定靶点，使用所有靶点
        if target_indices is None:
            target_indices = self.target_indices
        
        # 获取化合物嵌入
        compound_embedding = self.embeddings[compound_idx]
        
        # 计算与所有靶点的相似度
        similarity_scores = []
        importance_scores = []
        
        for target_idx in target_indices:
            # 相似度分数
            target_embedding = self.embeddings[target_idx]
            similarity = torch.cosine_similarity(
                compound_embedding.unsqueeze(0), 
                target_embedding.unsqueeze(0)
            ).item()
            similarity_scores.append(similarity)
            
            # 重要性分数
            target_id = self.reverse_node_map.get(target_idx.item(), f"Target_{target_idx.item()}")
            importance = self.important_targets.get(target_id, 0.0)
            importance_scores.append(importance)
        
        # 转换为张量
        similarity_scores = torch.tensor(similarity_scores)
        importance_scores = torch.tensor(importance_scores)
        
        # 归一化重要性分数
        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.max()
        
        # 使用学习到的权重计算组合分数
        combined_scores = (
            self.embedding_weight * similarity_scores +
            self.importance_weight * importance_scores
        )
        
        # 创建靶点-分数对
        targets_with_scores = list(zip(target_indices.tolist(), combined_scores.tolist()))
        
        # 按分数降序排序
        targets_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回顶部K个
        if top_k:
            return targets_with_scores[:top_k]
        
        return targets_with_scores
    
    def train_weights(self, validated_pairs, epochs=100):
        """
        基于已验证的化合物-靶点对训练权重
        
        Args:
            validated_pairs: (comp_id, target_id) 元组列表
            epochs: 训练轮数
        """
        print("训练动态权重...")
        
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            
            for comp_id, target_id in validated_pairs:
                # 获取索引
                comp_idx = self.node_map.get(comp_id)
                target_idx = self.node_map.get(target_id)
                
                if comp_idx is None or target_idx is None:
                    continue
                
                # 将索引转为张量
                comp_idx = torch.tensor(comp_idx)
                target_idx = torch.tensor(target_idx)
                
                # 计算此化合物的所有目标优先级
                all_scores = self.compute_priorities(comp_idx, self.target_indices)
                all_indices = [t[0] for t in all_scores]
                
                # 找到已验证靶点的排名
                try:
                    # 在排名列表中找到目标的位置
                    target_rank = all_indices.index(target_idx.item())
                    
                    # 排名越低，损失越高
                    loss = torch.tensor(target_rank, dtype=torch.float)
                    
                    # 如果目标在前10名，计为正确预测
                    if target_rank < 10:
                        correct_predictions += 1
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # 重新归一化权重
                    self._normalize_weights()
                    
                    total_loss += loss.item()
                except ValueError:
                    # 如果目标不在排名列表中
                    print(f"警告: 目标 {target_id} 不在排名列表中")
            
            # 计算准确率
            accuracy = correct_predictions / len(validated_pairs) if validated_pairs else 0
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss / len(validated_pairs) if validated_pairs else 0:.4f}, "
                      f"Accuracy@10: {accuracy:.4f}, "
                      f"Weights: Embedding={self.embedding_weight.item():.4f}, "
                      f"Importance={self.importance_weight.item():.4f}")
    
    def get_weights(self):
        """
        获取当前权重
        
        Returns:
            (embedding_weight, importance_weight) 元组
        """
        return (self.embedding_weight.item(), self.importance_weight.item())

def compute_all_tcm_targets(embeddings, node_map, reverse_node_map, important_targets, 
                         compound_indices, target_indices, validated_pairs=None, 
                         embedding_weight=0.6, importance_weight=0.4):
    """
    为所有TCM化合物计算靶点优先级
    
    Args:
        embeddings: 节点嵌入
        node_map: 节点ID到索引的映射
        reverse_node_map: 索引到节点ID的映射
        important_targets: 重要靶点字典
        compound_indices: 化合物索引
        target_indices: 靶点索引
        validated_pairs: 验证的化合物-靶点对 (可选，用于训练动态权重)
        embedding_weight: 嵌入相似度权重
        importance_weight: 靶点重要性权重
        
    Returns:
        所有化合物的靶点优先级字典
    """
    print("计算目标优先级...")
    
    # 初始化动态权重优先级排序器
    prioritizer = DynamicWeightTargetPrioritizer(
        embeddings, 
        node_map, 
        reverse_node_map, 
        target_indices, 
        important_targets,
        init_embedding_weight=embedding_weight,
        init_importance_weight=importance_weight
    )
    
    # 如果有验证数据，训练动态权重
    if validated_pairs:
        prioritizer.train_weights(validated_pairs, epochs=50)
    
    # 获取最终权重
    final_weights = prioritizer.get_weights()
    print(f"最终权重: 嵌入={final_weights[0]:.4f}, 重要性={final_weights[1]:.4f}")
    
    # 为所有化合物计算靶点优先级
    all_priorities = {}
    
    for idx in tqdm(compound_indices, desc="计算优先级"):
        # 获取化合物ID
        compound_id = reverse_node_map.get(idx.item(), f"Compound_{idx.item()}")
        
        # 计算目标优先级
        targets = prioritizer.compute_priorities(idx, target_indices)
        
        # 存储结果
        all_priorities[compound_id] = targets
    
    return all_priorities
