"""
验证数据处理模块

此模块负责加载和处理已验证的化合物-靶点相互作用数据，用于模型评估和优化。
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict

def load_validated_interactions(file_path):
    """
    加载已验证的化合物-靶点相互作用
    
    参数:
        file_path: 验证相互作用CSV文件路径
        
    返回:
        (compound_id, target_id) 元组列表
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 验证数据文件 {file_path} 不存在")
        return []
    
    try:
        # 加载CSV数据
        validated_data = pd.read_csv(file_path)
        
        # 确保必需列存在
        required_columns = ['compound_id', 'target_id']
        missing_columns = [col for col in required_columns if col not in validated_data.columns]
        
        if missing_columns:
            print(f"警告: 验证数据中缺少必需列: {missing_columns}")
            return []
        
        # 处理可能的NaN值
        validated_data = validated_data.dropna(subset=required_columns)
        
        # 提取为元组列表
        validated_pairs = list(zip(validated_data['compound_id'], validated_data['target_id']))
        
        # 如果有可选的'label'列，也包含它（用于阳性/阴性样本）
        if 'label' in validated_data.columns:
            validated_pairs_with_label = list(zip(
                validated_data['compound_id'], 
                validated_data['target_id'],
                validated_data['label']
            ))
            print(f"已加载带标签的验证相互作用数据: {len(validated_pairs_with_label)} 对")
            return validated_pairs_with_label
        
        print(f"已加载验证相互作用数据: {len(validated_pairs)} 对")
        return validated_pairs
    
    except Exception as e:
        print(f"加载验证相互作用数据时出错: {e}")
        return []

def split_validation_data(validated_pairs, test_ratio=0.2, random_seed=42):
    """
    将验证数据拆分为训练集和测试集
    
    参数:
        validated_pairs: 验证的化合物-靶点对
        test_ratio: 测试集比例
        random_seed: 随机种子
        
    返回:
        训练集和测试集对
    """
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 随机打乱数据
    indices = np.random.permutation(len(validated_pairs))
    test_size = int(len(validated_pairs) * test_ratio)
    
    # 拆分索引
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 创建训练集和测试集
    train_pairs = [validated_pairs[i] for i in train_indices]
    test_pairs = [validated_pairs[i] for i in test_indices]
    
    print(f"数据拆分: {len(train_pairs)} 训练样本, {len(test_pairs)} 测试样本")
    
    return train_pairs, test_pairs

def get_compound_target_coverage(validated_pairs, all_compounds, all_targets):
    """
    计算验证数据的化合物和靶点覆盖率
    
    参数:
        validated_pairs: 验证的化合物-靶点对
        all_compounds: 所有化合物ID列表
        all_targets: 所有靶点ID列表
        
    返回:
        覆盖率统计和每个化合物/靶点的验证对数量
    """
    # 初始化计数器
    validated_compounds = set()
    validated_targets = set()
    compound_counts = defaultdict(int)
    target_counts = defaultdict(int)
    
    # 统计覆盖率
    for pair in validated_pairs:
        if len(pair) >= 2:  # 确保有compound_id和target_id
            comp_id, target_id = pair[0], pair[1]
            validated_compounds.add(comp_id)
            validated_targets.add(target_id)
            compound_counts[comp_id] += 1
            target_counts[target_id] += 1
    
    # 计算覆盖百分比
    compound_coverage = len(validated_compounds) / len(all_compounds) * 100 if all_compounds else 0
    target_coverage = len(validated_targets) / len(all_targets) * 100 if all_targets else 0
    
    # 创建统计结果
    stats = {
        'total_pairs': len(validated_pairs),
        'unique_compounds': len(validated_compounds),
        'unique_targets': len(validated_targets),
        'compound_coverage_percent': compound_coverage,
        'target_coverage_percent': target_coverage,
        'avg_targets_per_compound': len(validated_pairs) / len(validated_compounds) if validated_compounds else 0,
        'avg_compounds_per_target': len(validated_pairs) / len(validated_targets) if validated_targets else 0
    }
    
    return stats, compound_counts, target_counts
