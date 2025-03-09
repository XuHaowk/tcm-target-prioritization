# 新建文件 src/models/ensemble.py

import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

class TCMTargetEnsemble:
    """集成多个模型的靶点优先级排序系统"""
    
    def __init__(self, gnn_model, importance_scores, validated_pairs=None):
        """
        初始化集成模型
        
        Args:
            gnn_model: 训练好的GNN模型
            importance_scores: 靶点重要性评分字典
            validated_pairs: 验证的化合物-靶点对列表 (可选)
        """
        self.gnn_model = gnn_model
        self.importance_scores = importance_scores
        
        # 初始化机器学习元模型
        self.meta_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        
        # 如果提供了验证数据，训练元模型
        if validated_pairs:
            self.train_meta_model(validated_pairs)
    
    def extract_features(self, comp_idx, target_idx, embeddings):
        """为元模型提取特征
        
        Args:
            comp_idx: 化合物节点索引
            target_idx: 靶点节点索引
            embeddings: 模型生成的嵌入
            
        Returns:
            特征向量
        """
        # 获取嵌入
        comp_emb = embeddings[comp_idx]
        target_emb = embeddings[target_idx]
        
        # 计算相似度特征
        cos_sim = torch.cosine_similarity(comp_emb.unsqueeze(0), target_emb.unsqueeze(0))
        euclidean_dist = torch.norm(comp_emb - target_emb)
        
        # 获取目标重要性
        target_id = self.reverse_node_map.get(target_idx, f"target_{target_idx}")
        importance = self.importance_scores.get(target_id, 0.0)
        
        # 组合所有特征
        features = [
            cos_sim.item(),
            euclidean_dist.item(),
            importance,
            comp_emb.mean().item(),
            comp_emb.std().item(),
            target_emb.mean().item(),
            target_emb.std().item()
        ]
        
        # 添加嵌入元素作为特征 (可选，取决于嵌入维度)
        # 如果嵌入很大，可以只使用前几个元素
        max_emb_features = 20
        emb_dim = min(comp_emb.shape[0], max_emb_features)
        features.extend(comp_emb[:emb_dim].tolist())
        features.extend(target_emb[:emb_dim].tolist())
        
        return np.array(features)
    
    def train_meta_model(self, validated_pairs):
        """训练元模型
        
        Args:
            validated_pairs: (comp_id, target_id, label)元组列表
                             label为1表示有效结合，0表示不结合
        """
        # 准备训练数据
        X = []
        y = []
        
        # 确保模型处于评估模式
        self.gnn_model.eval()
        
        # 获取嵌入
        with torch.no_grad():
            # 假设我们有这些数据
            data = ...  # 图数据
            embeddings = self.gnn_model(data.x, data.edge_index)
        
        # 为每个验证的对提取特征
        for comp_id, target_id, label in validated_pairs:
            comp_idx = self.node_map.get(comp_id)
            target_idx = self.node_map.get(target_id)
            
            if comp_idx is None or target_idx is None:
                continue
            
            # 提取特征
            features = self.extract_features(comp_idx, target_idx, embeddings)
            
            X.append(features)
            y.append(float(label))
        
        # 训练元模型
        if X and y:
            X = np.array(X)
            y = np.array(y)
            self.meta_model.fit(X, y)
            print(f"元模型训练完成，使用{len(X)}个样本")
    
    def predict_score(self, comp_idx, target_idx, embeddings):
        """使用元模型预测结合评分
        
        Args:
            comp_idx: 化合物节点索引
            target_idx: 靶点节点索引
            embeddings: 模型嵌入
        
        Returns:
            预测评分
        """
        # 提取特征
        features = self.extract_features(comp_idx, target_idx, embeddings)
        
        # 用元模型预测
        features = features.reshape(1, -1)  # 重塑为单个样本
        score = self.meta_model.predict(features)[0]
        
        return score
    
    def prioritize_targets(self, comp_idx, all_targets, embeddings):
        """为给定化合物对所有目标进行优先级排序
        
        Args:
            comp_idx: 化合物节点索引
            all_targets: 所有目标索引的列表
            embeddings: 模型嵌入
        
        Returns:
            (target_idx, score)元组列表，按评分降序排序
        """
        results = []
        
        for target_idx in all_targets:
            score = self.predict_score(comp_idx, target_idx, embeddings)
            results.append((target_idx, score))
        
        # 按评分排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def save(self, path):
        """保存元模型
        
        Args:
            path: 保存路径
        """
        joblib.dump(self.meta_model, path)
    
    def load(self, path):
        """加载元模型
        
        Args:
            path: 加载路径
        """
        self.meta_model = joblib.load(path)
