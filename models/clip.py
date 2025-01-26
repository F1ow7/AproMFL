import torch
import torch.nn as nn
import open_clip
from PIL import Image
import torch.nn.functional as F

class clipEncoder():
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',model_name='ViT-L-14', pretrained='laion2b_s32b_b82k'):
        self.model, _, self.transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
    def img_enc(self, image_tensor):
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
          
            
        return image_embedding
    def txt_enc(self, texts, mul = True):
       
        if mul:
            batch_texts_tok = self.tokenizer([text for i, ts in enumerate(texts) for text in ts]).to(self.device)
        else:
            batch_texts_tok = self.tokenizer(texts).to(self.device) 
        with torch.no_grad():
            text_embedding = self.model.encode_text(batch_texts_tok)
            # text_embedding = F.normalize(text_embedding, dim=-1)
        return text_embedding 

class ImageEncoder(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',model_name='ViT-L-14', pretrained='laion2b_s32b_b82k'):
        super(ImageEncoder, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()  # 设置模型为评估模式
       
        self.device = device
        self.model = self.model.to(self.device)
    
    def forward(self, image_tensor):
        
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
            image_embedding = F.normalize(image_embedding, dim=-1)
            
        return image_embedding


class TextEncoder(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', model_name='ViT-L-14', pretrained='laion2b_s32b_b82k'):
        super(TextEncoder, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()  
        self.tokenizer = open_clip.get_tokenizer(model_name)
       
        self.device = device
        self.model = self.model.to(self.device)
        
    
    def forward(self, texts, mul = True):
        
        if mul:
            batch_texts_tok = self.tokenizer([text for i, ts in enumerate(texts) for text in ts]).to(self.device)
        else:
            batch_texts_tok = self.tokenizer(texts).to(self.device)
       
        with torch.no_grad():
            text_embedding = self.model.encode_text(batch_texts_tok)
            text_embedding = F.normalize(text_embedding, dim=-1)
        return text_embedding

# 使用实例
if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_encoder = ImageEncoder(device=device)
    text_encoder = TextEncoder(device=device)
    
    
    image = Image.open("path_to_image.jpg")  # 输入图像路径
    text = "This is a sample text for encoding."

    
    image_embedding = image_encoder(image)
    text_embedding = text_encoder(text)
    
    print("Image Embedding Shape:", image_embedding.shape)
    print("Text Embedding Shape:", text_embedding.shape)
