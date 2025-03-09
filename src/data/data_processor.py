"""
数据处理器基类

提供所有数据处理器共享的通用功能
"""

import pandas as pd
import os
import numpy as np

class DataProcessor:
    """
    数据处理器基类，提供通用的数据加载和处理方法
    """
    
    def __init__(self, file_path=None):
        """
        初始化数据处理器
        
        参数:
            file_path: 数据文件路径(可选)
        """
        self.file_path = file_path
        self.data = None
    
    def load_data(self, file_path=None):
        """
        从CSV文件加载数据
        
        参数:
            file_path: 数据文件路径(可选，如果不提供则使用初始化时的路径)
            
        返回:
            加载的DataFrame
        """
        # 使用提供的路径或默认路径
        path = file_path or self.file_path
        
        # 检查文件是否存在
        if not path or not os.path.exists(path):
            print(f"错误: 文件 {path} 不存在")
            return pd.DataFrame()
        
        try:
            # 加载CSV数据
            self.data = pd.read_csv(path)
            print(f"成功加载数据: {path}")
            return self.data
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return pd.DataFrame()
    
    def validate_columns(self, required_columns):
        """
        验证DataFrame是否包含所有必需的列
        
        参数:
            required_columns: 必需列名的列表
            
        返回:
            缺失列的列表
        """
        if self.data is None:
            return required_columns
        
        missing_columns = []
        for col in required_columns:
            if col not in self.data.columns:
                missing_columns.append(col)
        
        return missing_columns
    
    def fill_missing_columns(self, missing_columns):
        """
        用空值填充缺失的列
        
        参数:
            missing_columns: 缺失列名的列表
        """
        if self.data is None:
            self.data = pd.DataFrame()
        
        for col in missing_columns:
            self.data[col] = None
    
    def safe_convert_to_numeric(self, column_name, default_value=0.0):
        """
        安全地将列转换为数值类型
        
        参数:
            column_name: 要转换的列名
            default_value: 转换失败时的默认值
        """
        if self.data is None or column_name not in self.data.columns:
            return
        
        self.data[column_name] = pd.to_numeric(self.data[column_name], errors='coerce')
        self.data[column_name] = self.data[column_name].fillna(default_value)
