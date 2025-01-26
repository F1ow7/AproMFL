import torch
from torch import nn
from torch.nn import functional as F
    
class ImageHead(nn.Module):
    def __init__(self, y_dim, bit, norm=True, hid_num=[1024, 1024]):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        
        super(ImageHead, self).__init__()
        self.module_name = "img_head"
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
        
             

    def forward(self, x):
        out1 = self.fc(x)
        out = torch.tanh(out1)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out1, out

    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False
            
            
class TextHead(nn.Module):
    def __init__(self, y_dim, bit, norm=True, hid_num=[1024, 1024]):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        
        super(TextHead, self).__init__()
        self.module_name = "text_head"
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
        
             

    def forward(self, x):
        out1 = self.fc(x)
        out = torch.tanh(out1)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out1, out

    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False