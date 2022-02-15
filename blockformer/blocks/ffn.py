import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..builder import BLOCKS

@BLOCKS.register_module()
class FFNBlock(BaseModule):
    def __init__(self,
            in_channels=256,
            expansion_ratio=4,
            mode='both',
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            init_cfg=None
        ):
        super(FFNBlock, self).__init__(init_cfg)
        hidden_dim = int(in_channels * expansion_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            build_from_cfg(act_cfg, ACTIVATION_LAYERS),
            nn.Linear(hidden_dim, in_channels)
        )
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
        self.mode = mode
    
    def init_weights(self):
        pass
    
    def forward_single(self, v):
        identity = v
        v = self.norm(v)
        v = self.ffn(v)
        v = self.dropout(v)
        v = v + identity
        return v
    
    def forward(self, x, t):
        if self.mode == 'both' or self.mode == 'x':
            x = self.forward_single(x)
        if self.mode == 'both' or self.mode == 't':
            t = self.forward_single(t)
        return x, t
