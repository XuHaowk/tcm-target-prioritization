"""
图构建模块，用于中医靶点优先级排序

本模块负责从化合物-靶点和疾病-靶点关系数据构建图表示。
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

def build_graph(compound_target_data, disease_target_data, compound_features, 
                target_features, disease_features=None):
    """
    根据关系数据和特征构建图数据对象
    
    参数:
        compound_target_data: 化合物-靶点关系DataFrame
        disease_target_data: 疾病-靶点关系DataFrame
        compound_features: 化合物特征字典
        target_features: 靶点特征字典
        disease_features: 疾病特征字典(可选)
        
    返回:
        图数据对象、节点映射、反向节点映射
    """
    print("构建图...")
    
    # 应用列映射
    # 为化合物-靶点数据映射列
    comp_col = 'compound' if 'compound' in compound_target_data.columns else 'compound_id'
    target_col = 'target' if 'target' in compound_target_data.columns else 'target_id'
    relation_col = 'relation_type' if 'relation_type' in compound_target_data.columns else 'relation_type'
    
    print(f"使用列: {comp_col} → compound_id, {target_col} → target_id, {relation_col} → relation_type")
    
    # 创建合适的化合物-靶点关系
    if comp_col in compound_target_data.columns and target_col in compound_target_data.columns:
        # 提取唯一化合物和靶点
        unique_compounds = compound_target_data[comp_col].unique()
        unique_targets = compound_target_data[target_col].unique()
        
        # 如果为空则创建特征字典
        if not compound_features:
            print(f"为 {len(unique_compounds)} 个化合物创建特征")
            compound_features = {str(comp): torch.randn(256) for comp in unique_compounds}
        
        if not target_features:
            print(f"为 {len(unique_targets)} 个靶点创建特征")
            target_features = {str(target): torch.randn(256) for target in unique_targets}
    
    # 检查特征字典是否为空，如果需要则修复
    if not compound_features:
        print("警告: 未找到化合物特征。创建随机特征。")
        compound_features = {f"compound_{i}": torch.randn(256) for i in range(10)}
    
    if not target_features:
        print("警告: 未找到靶点特征。创建随机特征。")
        # 尝试从数据中提取靶点ID
        target_ids = set()
        if target_col in compound_target_data.columns:
            target_ids.update(compound_target_data[target_col].unique())
        if disease_target_data is not None and target_col in disease_target_data.columns:
            target_ids.update(disease_target_data[target_col].unique())
        
        # 如果未找到靶点，创建一些占位符
        if not target_ids:
            target_ids = [f"target_{i}" for i in range(10)]
            
        # 为所有靶点创建随机特征
        feature_dim = next(iter(compound_features.values())).shape[0]
        target_features = {str(target_id): torch.randn(feature_dim) for target_id in target_ids}
    
    if disease_features is None or not disease_features:
        print("警告: 未找到疾病特征。如需要将创建随机特征。")
        disease_features = {}
        disease_col = 'disease' if disease_target_data is not None and 'disease' in disease_target_data.columns else 'disease_id'
        if disease_target_data is not None and disease_col in disease_target_data.columns:
            disease_ids = disease_target_data[disease_col].unique()
            feature_dim = next(iter(compound_features.values())).shape[0]
            disease_features = {str(disease_id): torch.randn(feature_dim) for disease_id in disease_ids}
    
    # 提取所有节点ID
    compound_ids = list(compound_features.keys())
    target_ids = list(target_features.keys())
    disease_ids = list(disease_features.keys()) if disease_features else []
    
    # 创建节点ID到索引的映射
    node_map = {}
    idx = 0
    
    # 映射化合物节点
    for comp_id in compound_ids:
        node_map[comp_id] = idx
        idx += 1
    
    # 映射靶点节点
    for target_id in target_ids:
        node_map[target_id] = idx
        idx += 1
    
    # 映射疾病节点
    for disease_id in disease_ids:
        node_map[disease_id] = idx
        idx += 1
    
    # 创建反向映射
    reverse_node_map = {idx: node_id for node_id, idx in node_map.items()}
    
    # 确定特征维度
    feature_dims = []
    if compound_features:
        feature_dims.append(next(iter(compound_features.values())).shape[0])
    if target_features:
        feature_dims.append(next(iter(target_features.values())).shape[0])
    if disease_features:
        feature_dims.append(next(iter(disease_features.values())).shape[0])
        
    # 使用最大维度
    if not feature_dims:
        print("错误: 未找到特征。使用默认维度 256。")
        feature_dim = 256
    else:
        feature_dim = max(feature_dims)
    
    # 创建特征矩阵
    node_features = torch.zeros((len(node_map), feature_dim))
    
    # 用化合物特征填充特征矩阵
    for comp_id, comp_idx in node_map.items():
        if comp_id in compound_features:
            feat = compound_features[comp_id]
            # 确保特征具有正确的形状
            if len(feat) < feature_dim:
                # 如需要用零填充
                padded = torch.zeros(feature_dim)
                padded[:len(feat)] = feat
                feat = padded
            elif len(feat) > feature_dim:
                # 如果太长则截断
                feat = feat[:feature_dim]
            node_features[comp_idx, :] = feat
    
    # 用靶点特征填充特征矩阵
    for target_id, target_idx in node_map.items():
        if target_id in target_features:
            feat = target_features[target_id]
            # 确保特征具有正确的形状
            if len(feat) < feature_dim:
                padded = torch.zeros(feature_dim)
                padded[:len(feat)] = feat
                feat = padded
            elif len(feat) > feature_dim:
                feat = feat[:feature_dim]
            node_features[target_idx, :] = feat
    
    # 用疾病特征填充特征矩阵
    if disease_features:
        for disease_id, disease_idx in node_map.items():
            if disease_id in disease_features:
                feat = disease_features[disease_id]
                # 确保特征具有正确的形状
                if len(feat) < feature_dim:
                    padded = torch.zeros(feature_dim)
                    padded[:len(feat)] = feat
                    feat = padded
                elif len(feat) > feature_dim:
                    feat = feat[:feature_dim]
                node_features[disease_idx, :] = feat
    
    # 准备边数据
    edge_indices = []
    edge_types = []
    edge_weights = []
    
    # 定义安全获取列数据的函数
    def safe_get_column(df, col_name, default_col_names=None):
        """安全获取列数据，如果需要尝试替代列名"""
        if col_name in df.columns:
            return df[col_name]
        elif default_col_names:
            for alt_name in default_col_names:
                if alt_name in df.columns:
                    print(f"使用列 '{alt_name}' 替代 '{col_name}'")
                    return df[alt_name]
        # 如果未找到合适的列，返回空Series
        print(f"警告: 未找到列 '{col_name}'。使用第一列作为备选。")
        if len(df.columns) > 0:
            return df.iloc[:, 0]
        return pd.Series([])
    
    # 处理化合物-靶点边
    print(f"处理化合物-靶点关系... ({len(compound_target_data)} 行)")
    compound_target_count = 0
    if not compound_target_data.empty:
        for _, row in compound_target_data.iterrows():
            # 尝试获取化合物和靶点ID
            comp_id = str(row.get(comp_col, f"compound_{compound_target_count}"))
            target_id = str(row.get(target_col, f"target_{compound_target_count}"))
            
            # 如果节点不在映射中则跳过
            if comp_id not in node_map or target_id not in node_map:
                continue
                
            comp_idx = node_map[comp_id]
            target_idx = node_map[target_id]
            
            # 添加边（双向）
            edge_indices.append([comp_idx, target_idx])
            edge_indices.append([target_idx, comp_idx])
            
            # 边类型（0 = 化合物-靶点）
            edge_types.append(0)
            edge_types.append(0)
            
            # 边权重（默认1.0）
            confidence = row.get('confidence_score', 1.0)
            if not isinstance(confidence, (int, float)):
                confidence = 1.0
            edge_weights.append(float(confidence))
            edge_weights.append(float(confidence))
            
            compound_target_count += 1
    
    # 处理疾病-靶点边
    disease_target_count = 0
    if disease_target_data is not None and not disease_target_data.empty:
        print(f"处理疾病-靶点关系... ({len(disease_target_data)} 行)")
        # 尝试识别疾病和靶点列
        disease_col = 'disease' if 'disease' in disease_target_data.columns else 'disease_id'
        
        for _, row in disease_target_data.iterrows():
            # 尝试获取疾病和靶点ID
            disease_id = str(row.get(disease_col, f"disease_{disease_target_count}"))
            target_id = str(row.get(target_col, f"target_{disease_target_count}"))
            
            # 如果节点不在映射中则跳过
            if disease_id not in node_map or target_id not in node_map:
                continue
                
            disease_idx = node_map[disease_id]
            target_idx = node_map[target_id]
            
            # 添加边（双向）
            edge_indices.append([disease_idx, target_idx])
            edge_indices.append([target_idx, disease_idx])
            
            # 边类型（1 = 疾病-靶点）
            edge_types.append(1)
            edge_types.append(1)
            
            # 边权重（如果可用则使用重要性）
            importance = row.get('importance_score', 1.0)
            if not isinstance(importance, (int, float)):
                importance = 1.0
            edge_weights.append(float(importance))
            edge_weights.append(float(importance))
            
            disease_target_count += 1
    
    # 如果找不到边，创建随机连接
    if not edge_indices:
        print("警告: 图中未找到边。创建随机关系结构。")
        
        # 确保我们有足够的化合物和靶点
        if len(compound_ids) > 0 and len(target_ids) > 0:
            print(f"使用 {len(compound_ids)} 个化合物和 {len(target_ids)} 个靶点创建随机关系")
            # 创建每个化合物到至少一个随机靶点的边
            for comp_id in compound_ids:
                comp_idx = node_map[comp_id]
                # 选择一个随机靶点
                for target_id in np.random.choice(target_ids, size=min(3, len(target_ids)), replace=False):
                    target_idx = node_map[target_id]
                    
                    # 添加双向边
                    edge_indices.append([comp_idx, target_idx])
                    edge_indices.append([target_idx, comp_idx])
                    
                    # 边类型和权重
                    edge_types.extend([0, 0])  # 化合物-靶点类型
                    edge_weights.extend([1.0, 1.0])
                    
                    compound_target_count += 1
        else:
            # 创建一些自循环
            print("创建自循环边作为最小图结构")
            for i in range(len(node_map)):
                edge_indices.append([i, i])
                edge_types.append(2)  # 自循环类型
                edge_weights.append(1.0)
    
    # 从边数据创建PyTorch张量
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # 打印图统计信息
    print(f"图统计信息:")
    print(f"  节点: {len(node_map)} ({len(compound_ids)} 化合物, {len(target_ids)} 靶点, {len(disease_ids)} 疾病)")
    print(f"  边: {len(edge_indices)} ({compound_target_count} 化合物-靶点, {disease_target_count} 疾病-靶点)")
    
    # 创建PyTorch Geometric Data对象
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight
    )
    
    # 存储每种节点类型的索引
    compound_indices = [node_map[comp_id] for comp_id in compound_ids if comp_id in node_map]
    target_indices = [node_map[target_id] for target_id in target_ids if target_id in node_map]
    disease_indices = [node_map[disease_id] for disease_id in disease_ids if disease_id in node_map]
    
    data.compound_indices = torch.tensor(compound_indices)
    data.target_indices = torch.tensor(target_indices)
    if disease_indices:
        data.disease_indices = torch.tensor(disease_indices)
    
    return data, node_map, reverse_node_map
