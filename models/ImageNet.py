import torch
from torch import nn
from torch.nn import functional as F
from models.clip import ImageEncoder

class ImageNet(nn.Module):
    def __init__(self, y_dim, bit, class_num=None, norm=False, hid_num=[1024, 1024], c_hid = [256,128]):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        
        super(ImageNet, self).__init__()
        # self.encoder = ImageEncoder()
        self.module_name = "img_model"
        hid = len(hid_num)
        if hid == 0:
            raise ValueError("Error: hid cannot be 0, please provide a valid hidden layer size.")
        modules = []
        pre = y_dim
        for i in range(hid):
            modules += [nn.Linear(pre, hid_num[i]), nn.ReLU(inplace=True)]
            pre = hid_num[i]
        modules += [nn.Linear(pre, bit)]
        
        self.fc = nn.Sequential(*modules)
        #self.apply(weights_init)
        self.norm = norm
        
        self.class_num = class_num
        self.classifier = None
        if class_num != 'None' and len(c_hid)>0:
            pre_c = bit
            mod = []
            for i in range(len(c_hid)):
                mod += [nn.Linear(pre_c, c_hid[i]), nn.ReLU(inplace=True)]
                pre_c = c_hid[i]
            mod += [nn.Linear(pre_c, class_num)]
            self.classifier = nn.Sequential(*mod)        

    def forward(self, x):
        out1 = self.fc(x)
        out = out1
        # out = torch.tanh(out1)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out1, out
    
    def classify(self, x):
        # Ensure the classifier exists before using it
        if self.classifier is None:
            raise ValueError("Classifier head is not defined. Set num_classes to a valid value when initializing the model.")
        
        # Use the forward pass of the main model to get embeddings
        # _, embedding = self.forward(x)
        
        # Pass the embedding through the classifier to get logits
        logits = self.classifier(x)
        
        return logits
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_grad(self):
        for p in self.parameters():
            p.requires_grad = True

# class ImageNet(nn.Module):
#     def __init__(self, code_len):
#         super(ImageNet, self).__init__()
#         self.fc1 = nn.Linear(4096, 4096)
#         self.fc_encode = nn.Linear(4096, code_len)

#         self.alpha = 1.0
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU(inplace=True)
#        # torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.1)  

#     def forward(self, x):

#         x = x.view(x.size(0), -1)

#         feat1 = self.relu(self.fc1(x))
#         #feat1 = feat1 + self.relu(self.fc2(self.dropout(feat1)))
#         hid = self.fc_encode(self.dropout(feat1))
#         code = torch.tanh(hid)

#         return code
