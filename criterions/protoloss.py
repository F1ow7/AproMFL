import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLoss(nn.Module):
    def __init__(self, prototypes):
                  prototypes['text']: Tensor (20, embedding_dim)
        
        super(PrototypeLoss, self).__init__()
        self.image_prototypes = torch.stack(prototypes['image'])  
        self.text_prototypes = torch.stack(prototypes['text'])    
        self.image_prototypes = F.normalize(self.image_prototypes, p=2, dim=1)
        self.text_prototypes = F.normalize(self.text_prototypes, p=2, dim=1)

    def forward(self, image_embedding, text_embedding,re =True):
       
        if re:
            image_embedding = image_embedding.repeat_interleave(5, dim=0)

        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)

     
        image_similarities = torch.matmul(image_embedding, self.image_prototypes.T)  
        image_probs = F.softmax(image_similarities, dim=1)  

        
        text_similarities = torch.matmul(text_embedding, self.text_prototypes.T)  
        text_probs = F.softmax(text_similarities, dim=1) 

        m = 0.5 * (image_probs + text_probs)
        js_divergence = 0.5 * (F.kl_div(image_probs.log(), m, reduction='batchmean') +
                               F.kl_div(text_probs.log(), m, reduction='batchmean'))

        return js_divergence
