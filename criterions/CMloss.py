import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, margin1=0.8, margin2=0.2):
        """
        初始化对比损失函数。
        :param margin: 正样本与负样本之间的最小距离（用于构造边界）。
        """
        super(CrossModalContrastiveLoss, self).__init__()
        self.margin_pos = margin1
        self.margin_neg = margin2

    def forward(self, image_embeddings, text_embeddings):

        similarity_matrix = F.cosine_similarity(image_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=2)
        
       
        batch_size = image_embeddings.size(0)
        labels = torch.eye(batch_size).to(image_embeddings.device)  # 匹配对的标签为1，非匹配对的标签为0
        
        
        positive_loss = labels * F.relu(self.margin_pos - similarity_matrix)  
        positive_loss = positive_loss.sum() / batch_size
        
        
        negative_loss = (1-labels) * F.relu(similarity_matrix - self.margin_neg)  # 负样本间距离要大于 margin
        negative_loss = negative_loss.sum() / (1-labels).sum()
        
        
        total_loss = positive_loss + negative_loss
        
        return total_loss
