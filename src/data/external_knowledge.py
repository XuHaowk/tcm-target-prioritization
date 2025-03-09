# 新建文件 src/data/external_knowledge.py

import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def get_compound_fingerprints(smiles_dict, radius=2, nBits=1024):
    """计算化合物的Morgan指纹
    
    Args:
        smiles_dict: 化合物ID到SMILES字符串的字典
        radius: 指纹半径
        nBits: 指纹比特数
    
    Returns:
        化合物ID到指纹向量的字典
    """
    fingerprints = {}
    
    for comp_id, smiles in smiles_dict.items():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
                fingerprints[comp_id] = torch.tensor(np.array(fp))
            else:
                print(f"警告: 无法解析SMILES: {smiles}")
                # 使用零向量作为后备
                fingerprints[comp_id] = torch.zeros(nBits)
        except Exception as e:
            print(f"计算指纹时出错 {comp_id}: {e}")
            fingerprints[comp_id] = torch.zeros(nBits)
    
    return fingerprints

def get_protein_features(uniprot_ids, feature_file=None):
    """获取蛋白质特征，如二级结构、功能域等
    
    Args:
        uniprot_ids: UniProt ID列表
        feature_file: 包含预计算特征的文件（可选）
    
    Returns:
        UniProt ID到特征向量的字典
    """
    protein_features = {}
    
    if feature_file:
        # 从文件加载预计算特征
        try:
            df = pd.read_csv(feature_file)
            for _, row in df.iterrows():
                uniprot_id = row['uniprot_id']
                if uniprot_id in uniprot_ids:
                    # 假设其余列是特征
                    features = torch.tensor(row.values[1:], dtype=torch.float)
                    protein_features[uniprot_id] = features
        except Exception as e:
            print(f"加载蛋白质特征时出错: {e}")
    
    # 对于缺失的蛋白质，使用零向量（真实应用中应该用API获取）
    feature_dim = next(iter(protein_features.values())).shape[0] if protein_features else 100
    for uniprot_id in uniprot_ids:
        if uniprot_id not in protein_features:
            protein_features[uniprot_id] = torch.zeros(feature_dim)
    
    return protein_features

def integrate_external_knowledge(node_features, node_ids, node_types, compound_smiles=None, protein_features_file=None):
    """将外部知识整合到节点特征中
    
    Args:
        node_features: 现有节点特征
        node_ids: 节点ID列表
        node_types: 节点类型列表
        compound_smiles: 化合物ID到SMILES的字典（可选）
        protein_features_file: 蛋白质特征文件（可选）
    
    Returns:
        增强的节点特征
    """
    # 复制原始特征
    enhanced_features = node_features.clone()
    
    # 如果提供了SMILES数据，计算化合物指纹
    if compound_smiles:
        compound_fps = get_compound_fingerprints(compound_smiles)
        
        # 获取蛋白质ID（假设它们是UniProt ID）
        protein_ids = [node_ids[i] for i, t in enumerate(node_types) if t == 'target']
        
        # 获取蛋白质特征
        protein_feats = get_protein_features(protein_ids, protein_features_file)
        
        # 合并特征
        for i, (node_id, node_type) in enumerate(zip(node_ids, node_types)):
            if node_type == 'compound' and node_id in compound_fps:
                # 将化合物指纹与现有特征连接
                fp = compound_fps[node_id]
                if enhanced_features.shape[1] < fp.shape[0]:
                    # 如果指纹维度更大，调整节点特征
                    padded = torch.zeros(enhanced_features.shape[0], fp.shape[0])
                    padded[:, :enhanced_features.shape[1]] = enhanced_features
                    enhanced_features = padded
                    
                # 合并特征
                enhanced_features[i, :fp.shape[0]] = fp
                
            elif node_type == 'target' and node_id in protein_feats:
                # 类似地合并蛋白质特征
                pf = protein_feats[node_id]
                if enhanced_features.shape[1] < pf.shape[0]:
                    padded = torch.zeros(enhanced_features.shape[0], pf.shape[0])
                    padded[:, :enhanced_features.shape[1]] = enhanced_features
                    enhanced_features = padded
                
                enhanced_features[i, :pf.shape[0]] = pf
    
    return enhanced_features
