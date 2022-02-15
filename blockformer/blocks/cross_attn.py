import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, ATTENTION
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..builder import BLOCKS

@BLOCKS.register_module()
class CrossAttentionBlock(BaseModule):
    def __init__(self,
            attn_cfg=dict(),
            norm_cfg=dict(type='LN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            init_cfg=None
        ):
        super(CrossAttentionBlock, self).__init__(init_cfg)
        self.attn = build_from_cfg(attn_cfg, ATTENTION)
        self.norm = build_norm_layer(norm_cfg, self.attn.v_dim)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)

    def forward(self, x, t):
        B, H, W, C = x.shape
        x = x.view(B, H*W, C) 
                
        identity = t
        t = self.norm(t)
        t = self.attn(t, x, x)
        t = self.dropout(t)
        t = t + identity
        
        x = x.view(B, H, W, C)
        return x, t
