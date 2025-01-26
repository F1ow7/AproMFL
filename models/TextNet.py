import torch
from torch import nn
from torch.nn import functional as F
from models.clip import TextEncoder

class TextNet(nn.Module):
    def __init__(self, y_dim, bit, class_num=None, norm=False, hid_num=[1024, 1024], c_hid = [256,128]):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TextNet, self).__init__()
        # self.encoder = TextEncoder()
        self.module_name = "txt_model"
        
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
        return out1,out
    
    def classify(self, x):
        
        if self.classifier is None:
            raise ValueError("Classifier head is not defined. Set num_classes to a valid value when initializing the model.")
        
    
        logits = self.classifier(x)
        
        return logits
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze_grad(self):
        for p in self.parameters():
            p.requires_grad = True
