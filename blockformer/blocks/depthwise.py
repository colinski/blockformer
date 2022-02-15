import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..builder import BLOCKS

@BLOCKS.register_module()
class DepthwiseBlock(BaseModule):
    def __init__(self,
            in_channels=256,
            expansion_ratio=4,
            mode='x',
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='BN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            init_cfg=None
        ):
        super(DepthwiseBlock, self).__init__(init_cfg)
        hidden_dim = int(in_channels * expansion_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, stride=1, groups=in_channels),
            build_from_cfg(act_cfg, ACTIVATION_LAYERS),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, padding=0, stride=1, groups=1),
        )
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
        self.mode = mode
    
    def init_weights(self):
        pass
    
    def forward_single(self, v):
        v = torch.einsum('BHWC -> BCHW', v)

        identity = v
        v = self.norm(v)
        v = self.conv(v)
        v = self.dropout(v)
        v = v + identity
        
        v = torch.einsum('BCHW -> BHWC', v)
        return v
    
    def forward(self, x, t):
        if self.mode == 'both' or self.mode == 'x':
            x = self.forward_single(x)
        if self.mode == 'both' or self.mode == 't':
            t = self.forward_single(t)
        return x, t

