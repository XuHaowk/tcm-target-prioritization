"""
数据库处理模块

此模块负责加载和处理化合物数据库信息，包括化合物特征向量。
"""

import pandas as pd
import numpy as np
import os
import json
import ast  # 用于安全地评估字符串表示的列表

def load_database(file_path):
    """
    从CSV文件加载化合物数据库数据
    
    参数:
        file_path: 数据库CSV文件路径
        
    返回:
        含有化合物数据的DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 数据库文件 {file_path} 不存在")
        return pd.DataFrame(columns=['compound_id', 'feature_vector'])
    
    try:
        # 加载CSV数据
        db_data = pd.read_csv(file_path)
        
        # 确保必需列存在
        required_columns = ['compound_id', 'feature_vector']
        missing_columns = [col for col in required_columns if col not in db_data.columns]
        
        if missing_columns:
            print(f"警告: 数据库数据中缺少必需列: {missing_columns}")
            
            # 如果特征向量缺失，创建随机特征
            if 'feature_vector' in missing_columns:
                print("为化合物创建随机特征向量")
                # 使用固定的随机种子以保证可重复性
                np.random.seed(42)
                db_data['feature_vector'] = [
                    str(list(np.random.random(256))) for _ in range(len(db_data))
                ]
            
            # 如果compound_id缺失，创建占位符ID
            if 'compound_id' in missing_columns:
                print("创建占位符化合物ID")
                db_data['compound_id'] = [f"compound_{i}" for i in range(len(db_data))]
        
        # 验证特征向量格式
        if len(db_data) > 0:
            sample_feature = db_data['feature_vector'].iloc[0]
            try:
                # 尝试解析特征向量，确保格式正确
                parse_feature_vector(sample_feature)
            except Exception as e:
                print(f"警告: 特征向量格式可能不正确: {e}")
                print("尝试修复特征向量格式...")
                db_data['feature_vector'] = db_data['feature_vector'].apply(fix_feature_vector_format)
        
        # 输出摘要信息
        print(f"已加载数据库数据: {len(db_data)} 化合物")
        
        return db_data
    
    except Exception as e:
        print(f"加载数据库数据时出错: {e}")
        # 返回包含必需列的空DataFrame
        return pd.DataFrame(columns=['compound_id', 'feature_vector'])

def parse_feature_vector(feature_str):
    """
    安全地解析特征向量字符串
    
    参数:
        feature_str: 特征向量的字符串表示
        
    返回:
        解析后的特征向量列表
    """
    try:
        # 尝试使用ast.literal_eval解析（比eval更安全）
        feature_vector = ast.literal_eval(feature_str)
        if not isinstance(feature_vector, list):
            raise ValueError("特征向量必须是列表")
        return feature_vector
    except:
        # 如果失败，尝试使用json解析
        try:
            feature_vector = json.loads(feature_str)
            if not isinstance(feature_vector, list):
                raise ValueError("特征向量必须是列表")
            return feature_vector
        except:
            # 如果仍然失败，尝试另一种常见格式
            if feature_str.startswith('[') and feature_str.endswith(']'):
                try:
                    # 尝试将字符串转换为列表
                    values = feature_str.strip('[]').split(',')
                    return [float(v.strip()) for v in values]
                except:
                    pass
            
            raise ValueError(f"无法解析特征向量: {feature_str[:50]}...")

def fix_feature_vector_format(feature_str):
    """
    尝试修复特征向量的格式
    
    参数:
        feature_str: 特征向量的字符串表示
        
    返回:
        修复后的特征向量字符串
    """
    try:
        # 尝试解析
        feature_vector = parse_feature_vector(feature_str)
        # 如果成功，返回一致的格式
        return str(feature_vector)
    except:
        # 如果解析失败，创建随机向量
        print(f"无法解析特征，创建随机向量替代: {feature_str[:30]}...")
        np.random.seed(42)  # 固定随机种子
        return str(list(np.random.random(256)))

def extract_compound_features(db_data):
    """
    从数据库数据中提取化合物特征
    
    参数:
        db_data: 数据库DataFrame
        
    返回:
        化合物ID到特征向量的字典
    """
    compound_features = {}
    
    for _, row in db_data.iterrows():
        compound_id = row['compound_id']
        
        try:
            # 解析特征向量
            features = parse_feature_vector(row['feature_vector'])
            # 将列表转换为numpy数组
            features_array = np.array(features, dtype=np.float32)
            compound_features[compound_id] = features_array
        except Exception as e:
            print(f"处理化合物 {compound_id} 的特征时出错: {e}")
            # 使用随机特征作为后备
            np.random.seed(hash(compound_id) % 2**32)  # 使用化合物ID作为种子
            compound_features[compound_id] = np.random.random(256).astype(np.float32)
    
    return compound_features
