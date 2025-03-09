"""
疾病重要性处理模块

此模块负责加载和处理疾病-靶点重要性数据，用于优先级排序。
"""

import pandas as pd
import numpy as np
import os
from src.data.data_processor import DataProcessor

def load_disease_importance(file_path):
    """
    从CSV文件加载疾病重要性数据
    
    参数:
        file_path: 疾病重要性CSV文件路径
        
    返回:
        含有疾病重要性数据的DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 疾病重要性文件 {file_path} 不存在")
        return pd.DataFrame(columns=['target_id', 'importance_score'])
    
    try:
        # 加载CSV数据
        importance_data = pd.read_csv(file_path)
        
        # 确保必需列存在
        required_columns = ['target_id', 'importance_score']
        missing_columns = [col for col in required_columns if col not in importance_data.columns]
        
        if missing_columns:
            print(f"警告: 疾病重要性数据中缺少必需列: {missing_columns}")
            
            # 如果重要性分数缺失，创建随机分数
            if 'importance_score' in missing_columns:
                print("创建随机重要性分数")
                # 固定随机种子以保证可重复性
                np.random.seed(42)
                importance_data['importance_score'] = [
                    float(np.random.random()) for _ in range(len(importance_data))
                ]
                
            # 如果target_id缺失，创建占位符ID
            if 'target_id' in missing_columns:
                print("创建占位符靶点ID")
                importance_data['target_id'] = [f"target_{i}" for i in range(len(importance_data))]
        
        # 规范化重要性分数（确保在0到1之间）
        if 'importance_score' in importance_data.columns:
            # 处理无效值
            importance_data['importance_score'] = pd.to_numeric(
                importance_data['importance_score'], errors='coerce'
            )
            # 替换NaN值
            importance_data['importance_score'] = importance_data['importance_score'].fillna(0.0)
            
            # 如果有负值，进行偏移使所有值非负
            min_score = importance_data['importance_score'].min()
            if min_score < 0:
                importance_data['importance_score'] = importance_data['importance_score'] - min_score
                
            # 规范化到0-1
            max_score = importance_data['importance_score'].max()
            if max_score > 0:  # 避免除零
                importance_data['importance_score'] = importance_data['importance_score'] / max_score
        
        # 输出摘要信息
        print(f"已加载疾病重要性数据: {len(importance_data)} 靶点")
        if 'importance_score' in importance_data.columns:
            print(f"重要性分数范围: {importance_data['importance_score'].min()} - {importance_data['importance_score'].max()}")
        
        return importance_data
    
    except Exception as e:
        print(f"加载疾病重要性数据时出错: {e}")
        # 返回包含必需列的空DataFrame
        return pd.DataFrame(columns=['target_id', 'importance_score'])

def create_importance_dictionary(importance_data):
    """
    创建靶点重要性字典
    
    参数:
        importance_data: 疾病重要性DataFrame
        
    返回:
        靶点ID到重要性分数的字典
    """
    importance_dict = {}
    
    # 确保列存在
    if 'target_id' not in importance_data.columns or 'importance_score' not in importance_data.columns:
        print("警告: 创建重要性字典时缺少必需列")
        return importance_dict
        
    for _, row in importance_data.iterrows():
        target_id = row['target_id']
        importance = row['importance_score']
        
        # 转换为字符串以保证一致性
        if not isinstance(target_id, str):
            target_id = str(target_id)
            
        # 确保重要性是浮点数
        try:
            importance = float(importance)
        except:
            importance = 0.0
            
        importance_dict[target_id] = importance
    
    # 对于没有重要性分数的靶点，使用默认值
    if not importance_dict:
        print("警告: 重要性字典为空")
    
    return importance_dict

def merge_target_importance(target_list, importance_dict, default_value=0.0):
    """
    为完整的靶点列表合并重要性信息
    
    参数:
        target_list: 靶点ID列表
        importance_dict: 靶点重要性字典
        default_value: 缺失靶点的默认重要性
        
    返回:
        完整的靶点重要性字典
    """
    complete_importance = {}
    
    for target_id in target_list:
        # 转换为字符串以保证一致性
        if not isinstance(target_id, str):
            target_id = str(target_id)
            
        # 如果存在使用字典值，否则使用默认值
        importance = importance_dict.get(target_id, default_value)
        complete_importance[target_id] = importance
    
    return complete_importance
