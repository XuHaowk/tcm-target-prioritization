"""
知识图谱数据处理模块

此模块负责加载和处理知识图谱数据，包括化合物-靶点关系和疾病-靶点关系。
"""

import pandas as pd
import numpy as np
import os
from src.data.data_processor import DataProcessor

def load_knowledge_graph(file_path):
    """
    从CSV文件加载知识图谱数据
    
    参数:
        file_path: 知识图谱CSV文件路径
        
    返回:
        含有知识图谱数据的DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 知识图谱文件 {file_path} 不存在")
        return pd.DataFrame(columns=['compound_id', 'target_id', 'disease_id', 'relation_type'])
    
    # 加载CSV数据
    try:
        kg_data = pd.read_csv(file_path)
        
        # 确保必需列存在
        required_columns = ['compound_id', 'target_id', 'relation_type']
        missing_columns = [col for col in required_columns if col not in kg_data.columns]
        
        if missing_columns:
            print(f"警告: 知识图谱数据中缺少必需列: {missing_columns}")
            # 为缺失列添加空值
            for col in missing_columns:
                kg_data[col] = None
                
        # 如果不存在，添加'disease_id'列（用于疾病-靶点关系）
        if 'disease_id' not in kg_data.columns:
            kg_data['disease_id'] = None
        
        # 处理可能的NaN值
        kg_data = kg_data.fillna({
            'compound_id': 'unknown_compound',
            'target_id': 'unknown_target',
            'disease_id': 'unknown_disease',
            'relation_type': 'unknown_relation'
        })
        
        # 输出摘要信息
        print(f"已加载知识图谱数据: {len(kg_data)} 行")
        relation_types = kg_data['relation_type'].unique()
        print(f"关系类型: {relation_types}")
        
        return kg_data
    
    except Exception as e:
        print(f"加载知识图谱数据时出错: {e}")
        # 返回包含必需列的空DataFrame
        return pd.DataFrame(columns=['compound_id', 'target_id', 'disease_id', 'relation_type'])

def extract_compound_target_relations(kg_data):
    """
    从知识图谱数据中提取化合物-靶点关系
    
    参数:
        kg_data: 知识图谱DataFrame
        
    返回:
        仅包含化合物-靶点关系的DataFrame
    """
    # 提取化合物-靶点关系
    compound_target_data = kg_data[kg_data['relation_type'] == 'compound_target']
    
    # 检查是否有数据
    if len(compound_target_data) == 0:
        print("警告: 未找到化合物-靶点关系")
        
    return compound_target_data

def extract_disease_target_relations(kg_data):
    """
    从知识图谱数据中提取疾病-靶点关系
    
    参数:
        kg_data: 知识图谱DataFrame
        
    返回:
        仅包含疾病-靶点关系的DataFrame
    """
    # 提取疾病-靶点关系
    disease_target_data = kg_data[kg_data['relation_type'] == 'disease_target']
    
    # 检查是否有数据
    if len(disease_target_data) == 0:
        print("警告: 未找到疾病-靶点关系")
        
    return disease_target_data

def get_unique_entities(kg_data):
    """
    从知识图谱中获取唯一实体列表
    
    参数:
        kg_data: 知识图谱DataFrame
        
    返回:
        化合物、靶点和疾病ID的唯一列表
    """
    # 获取化合物-靶点关系
    compound_target_data = extract_compound_target_relations(kg_data)
    
    # 获取疾病-靶点关系
    disease_target_data = extract_disease_target_relations(kg_data)
    
    # 提取唯一ID
    compounds = list(set(compound_target_data['compound_id']))
    targets_from_compounds = list(set(compound_target_data['target_id']))
    
    targets_from_diseases = []
    diseases = []
    
    if len(disease_target_data) > 0:
        targets_from_diseases = list(set(disease_target_data['target_id']))
        diseases = list(set(disease_target_data['disease_id']))
    
    # 合并靶点列表（可能从两种关系中出现）
    all_targets = list(set(targets_from_compounds + targets_from_diseases))
    
    return compounds, all_targets, diseases
