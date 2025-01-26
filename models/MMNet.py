import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.nn as nn

from src.utils.tensor_utils import l2_normalize

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from models.ImageNet import ImageNet
from models.TextNet import TextNet
from models.clip import ImageEncoder, TextEncoder

class MMNet(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self, config):
        super(MMNet, self).__init__()

        self.config = config
        # self.embed_dim = config.embed_dim
        # if config.get('n_samples_inference', 0):
        #     self.n_embeddings = config.n_samples_inference
        # else:
        #     self.n_embeddings = 1

        # self.img_enc = ImageEncoder()
        # self.txt_enc = TextEncoder()
        
        self.image_ph = ImageNet(config.image_p.input_embedding, config.image_p.out_embedding, config.image_p.class_num, config.image_p.norm,config.image_p.hid_num, config.image_p.c_hid)
        self.text_ph = TextNet(config.text_p.input_embedding, config.text_p.out_embedding, config.text_p.class_num, config.text_p.norm,config.text_p.hid_num, config.text_p.c_hid)
        
                # Freeze the parameters of img_enc and txt_enc (no gradients)
        # for param in self.img_enc.parameters():
        #     param.requires_grad = False
        # for param in self.txt_enc.parameters():
        #     param.requires_grad = False

        # Allow gradients for the parameters of image_ph and text_ph
        for param in self.image_ph.parameters():
            param.requires_grad = True
        for param in self.text_ph.parameters():
            param.requires_grad = True
            

    def forward(self, images, sentences):
        # image_embedding = self.img_enc(images)
        # image_output = self.image_ph(image_embedding)
        image_output = self.image_ph(images)[1]
        
        # caption_embedding = self.txt_enc(sentences)  # sentences: [128,  seq_len], lengths: 128
        # caption_output = self.text_ph(caption_embedding)
        
        caption_output = self.text_ph(sentences)[1]
            # inputs = self.tokenizer(captions_word, padding=True, return_tensors='pt')
            # for a in inputs:
            #     inputs[a] = inputs[a].cuda()
            # caption_output = self.txt_enc(**inputs)
            # caption_output = {'embedding': l2_normalize(self.linear(caption_output['last_hidden_state'][:, 0, :]))}  # [bsz, 768]

        # return {
        #     'image_embedding': image_embedding,
        #     'image_output': image_output,
        #     'caption_embedding': caption_embedding,
        #     'caption_output': caption_output
        # }
        
        return {
            'image_embedding': images,
            'image_output': image_output,
            'caption_embedding': sentences,
            'caption_output': caption_output
        }

    # def image_forward(self, images):
    #     return self.image_ph(self.img_enc(images))

    # def text_forward(self, sentences, lengths):
    #     return self.text_ph(self.txt_enc(sentences, lengths))
    def image_forward(self, images):
        return self.image_ph(images)

    def text_forward(self, sentences, lengths):
        return self.text_ph(sentences, lengths)