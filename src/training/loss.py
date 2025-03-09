import torch
import torch.nn.functional as F
import numpy as np

def contrastive_loss(embeddings, positive_pairs, negative_pairs, margin=0.5):
    """
    对比损失函数，具有NaN值处理能力
    
    Args:
        embeddings: 节点嵌入
        positive_pairs: 正样本对索引
        negative_pairs: 负样本对索引
        margin: 间隔参数
    
    Returns:
        损失值
    """
    # 替换NaN值
    if torch.isnan(embeddings).any():
        print("警告: 嵌入中检测到NaN值。正在替换为零。")
        embeddings = torch.nan_to_num(embeddings, nan=0.0)
    
    # 提取正样本对的嵌入
    pos_emb1 = embeddings[positive_pairs[:, 0]]
    pos_emb2 = embeddings[positive_pairs[:, 1]]
    
    # 提取负样本对的嵌入
    neg_emb1 = embeddings[negative_pairs[:, 0]] 
    neg_emb2 = embeddings[negative_pairs[:, 1]]
    
    # 计算余弦相似度
    pos_score = F.cosine_similarity(pos_emb1, pos_emb2)
    neg_score = F.cosine_similarity(neg_emb1, neg_emb2)
    
    # 处理相似度中的NaN值
    pos_score = torch.nan_to_num(pos_score, nan=0.0)
    neg_score = torch.nan_to_num(neg_score, nan=0.0)
    
    # 计算损失并确保数值稳定性
    pos_loss = torch.mean(torch.clamp(margin - pos_score, min=0))
    neg_loss = torch.mean(torch.clamp(neg_score - (-margin), min=0))
    
    # 组合损失
    loss = pos_loss + neg_loss
    
    # 最终安全检查
    if torch.isnan(loss):
        print("警告: 尽管有保护措施，仍检测到NaN损失。使用后备损失。")
        return torch.tensor(0.1, requires_grad=True, device=embeddings.device)
    
    return loss

def structure_aware_loss(embeddings, edge_index, edge_weight=None, margin=0.5, neg_samples=5):
    """
    结构感知损失函数，利用图结构
    
    Args:
        embeddings: 节点嵌入
        edge_index: 边索引
        edge_weight: 边权重 (可选)
        margin: 间隔参数
        neg_samples: 每个节点的负样本数
    
    Returns:
        结构感知损失
    """
    # 检测并处理NaN值
    if torch.isnan(embeddings).any():
        embeddings = torch.nan_to_num(embeddings, nan=0.0)
    
    # 计算所有节点对的相似度矩阵
    sim_matrix = torch.mm(embeddings, embeddings.t())
    
    # 提取实际边的相似度
    src, dst = edge_index
    pos_scores = sim_matrix[src, dst]
    
    # 为每个源节点生成随机负样本
    neg_scores_list = []
    
    # 获取图中的节点数
    num_nodes = embeddings.shape[0]
    
    for s in src.unique():
        # 找出与源节点s相连的所有目标节点
        connected = dst[src == s]
        
        # 生成不与s连接的随机节点作为负样本
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[connected] = False
        mask[s] = False  # 排除自身
        
        # 可作为负样本的节点索引
        neg_candidates = torch.nonzero(mask).squeeze()
        
        # 如果有足够的负样本候选者
        if neg_candidates.numel() > 0:
            # 随机选择neg_samples个
            if neg_candidates.numel() > neg_samples:
                perm = torch.randperm(neg_candidates.numel())
                neg_dst = neg_candidates[perm[:neg_samples]]
            else:
                neg_dst = neg_candidates
            
            # 获取这些负样本对的相似度分数
            for d in neg_dst:
                neg_scores_list.append(sim_matrix[s, d])
    
    # 将负样本分数合并为张量
    if neg_scores_list:
        neg_scores = torch.stack(neg_scores_list)
    else:
        # 如果没有找到负样本，创建一个占位符
        neg_scores = torch.tensor([-1.0], device=embeddings.device)
    
    # 三元组损失: 确保正样本比负样本相似度高出至少margin
    pos_loss = torch.mean(torch.clamp(margin - pos_scores, min=0))
    neg_loss = torch.mean(torch.clamp(neg_scores - (-margin), min=0))
    
    # 权重损失 (如果提供了边权重)
    if edge_weight is not None:
        # 加权正样本损失
        weighted_pos_loss = torch.mean(edge_weight * torch.clamp(margin - pos_scores, min=0))
        # 结合权重
        return weighted_pos_loss + neg_loss
    
    return pos_loss + neg_loss

def combined_loss(embeddings, positive_pairs, negative_pairs, edge_index, edge_weight=None, margin=0.5, 
                alpha=0.7, beta=0.3, neg_samples=5):
    """
    结合对比损失和结构感知损失
    
    Args:
        embeddings: 节点嵌入
        positive_pairs: 正样本对索引
        negative_pairs: 负样本对索引
        edge_index: 边索引
        edge_weight: 边权重 (可选)
        margin: 间隔参数
        alpha: 对比损失权重
        beta: 结构损失权重
        neg_samples: 每个节点的负样本数
    
    Returns:
        组合损失
    """
    # 计算对比损失
    contrast_loss = contrastive_loss(embeddings, positive_pairs, negative_pairs, margin)
    
    # 计算结构感知损失
    struct_loss = structure_aware_loss(embeddings, edge_index, edge_weight, margin, neg_samples)
    
    # 组合损失
    combined = alpha * contrast_loss + beta * struct_loss
    
    # 安全检查
    if torch.isnan(combined):
        print("警告: 组合损失中检测到NaN。使用后备损失。")
        return torch.tensor(0.1, requires_grad=True, device=embeddings.device)
    
    return combined
