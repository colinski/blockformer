import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner.base_module import BaseModule
from ..builder import BLOCKS

#modified version of a copy-paste of PatchMerging
#implemented shifted window patch embeddings ala PVT
@BLOCKS.register_module()
class DownsampleBlock(BaseModule):
    def __init__(self,
            in_channels=256,
            out_channels=512,
            factor=2,
            norm_cfg=dict(type='LN'),
            init_cfg=None
        ):
        super(DownsampleBlock, self).__init__(init_cfg)
        self.factor = factor
        self.kernel_size = 2 * factor - 1 #shifted window
        self.stride = factor
        self.padding = factor - 1
        hidden_dim = in_channels * self.kernel_size**2
        
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        )
        self.norm = build_norm_layer(norm_cfg, hidden_dim)[1]
        self.x_downsample = nn.Linear(hidden_dim, out_channels)
        self.t_downsample = nn.Linear(in_channels, out_channels)
    
    def init_weights(self):
        pass
    
    def forward(self, x, t):
        B, H, W, C = x.shape
        KS, S = self.kernel_size, self.stride
        x = x.permute(0, 3, 1, 2) #B C H W
        x = self.unfold(x) #B C*KS**2 H/KS*W/KS
        x = x.view(B, C*KS*KS, H//S, W//S)
        x = x.permute(0, 2, 3, 1)  #B H/S W/S hidden_dim
        
        x = self.norm(x)
        x = self.x_downsample(x) #B H/2 W/2 out_channels
        t = self.t_downsample(t)
        return x, t
