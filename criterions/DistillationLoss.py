import torch
import torch.nn.functional as F
import torch.nn as nn
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha  

    def forward(self, student_output, teacher_output, T=1.0):
        """
        计算蒸馏损失（KL 散度）。
        :param student_output: 学生模型输出的 logits (batch_size, num_classes)
        :param teacher_output: 老师模型输出的 logits (batch_size, num_classes)
        :param T: 温度因子，通常 T > 1，用于软化 logits
        :return: 蒸馏损失
        """
     
        student_probs = F.softmax(student_output / T, dim=1)
        teacher_probs = F.softmax(teacher_output / T, dim=1)
        
        
        loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean') * (T * T)
        return loss