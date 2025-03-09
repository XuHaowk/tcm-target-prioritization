import torch
import numpy as np
from torch_geometric.utils import degree

def build_node_features(compound_features, target_features, disease_features=None):
    """
    构建节点特征
    
    Args:
        compound_features: 化合物特征字典 {compound_id: feature_vector}
        target_features: 靶点特征字典 {target_id: feature_vector}
        disease_features: 疾病特征字典 {disease_id: feature_vector}
        
    Returns:
        合并的节点特征张量，节点ID到索引的映射，索引到节点ID的反向映射
    """
    # 合并所有节点ID
    node_ids = list(compound_features.keys()) + list(target_features.keys())
    if disease_features:
        node_ids += list(disease_features.keys())
    
    # 创建节点ID到索引的映射
    node_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # 创建反向映射
    reverse_node_map = {idx: node_id for node_id, idx in node_map.items()}
    
    # 确定特征维度
    compound_dim = next(iter(compound_features.values())).shape[0]
    target_dim = next(iter(target_features.values())).shape[0]
    disease_dim = next(iter(disease_features.values())).shape[0] if disease_features else 0
    
    # 取最大维度
    feature_dim = max(compound_dim, target_dim, disease_dim)
    
    # 创建节点特征矩阵
    node_features = torch.zeros((len(node_ids), feature_dim))
    
    # 创建节点类型列表
    node_types = []
    
    # 填充特征矩阵
    for node_id, idx in node_map.items():
        if node_id in compound_features:
            features = compound_features[node_id]
            node_features[idx, :features.shape[0]] = features
            node_types.append('compound')
        elif node_id in target_features:
            features = target_features[node_id]
            node_features[idx, :features.shape[0]] = features
            node_types.append('target')
        elif disease_features and node_id in disease_features:
            features = disease_features[node_id]
            node_features[idx, :features.shape[0]] = features
            node_types.append('disease')
    
    return node_features, node_map, reverse_node_map, node_types

def enhance_node_features(node_features, node_types, data=None):
    """
    增强节点特征以提供更丰富的信息
    
    Args:
        node_features: 原始节点特征
        node_types: 每个节点的类型 (化合物/靶点/疾病)
        data: 图数据对象(可选，用于计算拓扑特征)
    
    Returns:
        增强的节点特征
    """
    # 复制原始特征
    enhanced_features = node_features.clone()
    
    # 1. 添加节点类型的one-hot编码
    node_type_onehot = torch.zeros(node_features.shape[0], 3)
    for i, node_type in enumerate(node_types):
        if node_type == 'compound':
            node_type_onehot[i, 0] = 1.0
        elif node_type == 'target':
            node_type_onehot[i, 1] = 1.0
        elif node_type == 'disease':
            node_type_onehot[i, 2] = 1.0
    
    # 2. 添加度中心性特征 - 节点连接数量提供结构信息
    degree_features = torch.zeros(node_features.shape[0], 1)
    if data is not None and hasattr(data, 'edge_index'):
        edge_index = data.edge_index
        # 计算每个节点的度
        deg = degree(edge_index[0], num_nodes=node_features.shape[0])
        # 归一化
        if deg.max() > 0:
            normalized_deg = deg / deg.max()
            degree_features = normalized_deg.unsqueeze(1)
    
    # 将所有特征连接起来
    enhanced_features = torch.cat([enhanced_features, node_type_onehot, degree_features], dim=1)
    
    # 3. 应用特征缩放以提高稳定性
    mean = enhanced_features.mean(dim=0)
    std = enhanced_features.std(dim=0) + 1e-8
    enhanced_features = (enhanced_features - mean) / std
    
    return enhanced_features

def normalize_features(node_features):
    """
    归一化特征以提高数值稳定性
    
    Args:
        node_features: 节点特征张量
        
    Returns:
        归一化的特征
    """
    # 替换NaN值
    if torch.isnan(node_features).any():
        node_features = torch.nan_to_num(node_features, nan=0.0)
    
    # 计算均值和标准差，忽略零值
    mask = node_features != 0
    if mask.any():
        mean = node_features[mask].mean()
        std = node_features[mask].std() + 1e-8
        
        # 只对非零值应用归一化
        normalized = node_features.clone()
        normalized[mask] = (node_features[mask] - mean) / std
        return normalized
    else:
        return node_features
