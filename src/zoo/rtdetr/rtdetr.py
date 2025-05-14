"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        shape = x.shape
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            shape = x.shape
            if len(shape) == 5:
                assert shape[1] == 2
                x = x.reshape((shape[0]*2, shape[2],shape[3],shape[4]))
            x = F.interpolate(x, size=[sz, sz])
            if len(shape)==5:
                ss4 = x.shape
                x = x.reshape((ss4[0]//2, 2, ss4[1],ss4[2],ss4[3]))
        if self.training and len(shape) == 5:
            x1 = x[:,0,...]
            x2 = x[:,1,...]
            o1 = self.backbone(x1)
            o2 = self.backbone(x2)
            e1 = self.encoder(o1)
            e2 = self.encoder(o2)
            x = self.decoder(e2,targets)
            x['aux_emb_upr'] = e2
            x['aux_emb_ori'] = e1
        else:
            x = self.backbone(x)
            x = self.encoder(x)        
            x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
