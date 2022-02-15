import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, ATTENTION
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..builder import BLOCKS


@BLOCKS.register_module()
class SeqAttentionBlock(BaseModule):
    def __init__(self,
            attn_cfg=dict(),
            norm_cfg=dict(type='LN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            x_position_cfg=dict(type='SinePositionalEncoding', dim=256),
            t_position_cfg=dict(type='SinePositionalEncoding', dim=256),
            mode='both',
            init_cfg=None
        ):
        super(SeqAttentionBlock, self).__init__(init_cfg)
        self.attn = build_from_cfg(attn_cfg, ATTENTION)
        self.norm = build_norm_layer(norm_cfg, self.attn.v_dim)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
        self.pos_embed_x = build_from_cfg()
        self.pos_embed_t = build_from_cfg()
        self.mode = mode
    
    def forward_single(self, y, pos_embed):
        identity = y
        y = self.norm(y)
        q = k = pos_embed(y)
        y = self.attn(q, k, y)
        y = self.dropout(y)
        y = y + identity
        return y

    def forward(self, x, t=None):
        if self.mode == 'both' or self.mode == 'x':
            x = x.view(B, H*W, C)
            x = self.forward_single(x, self.pos_embed_x)
            x = x.view(B, H, W, C)
        if self.mode == 'both' or self.mode == 't':
            t = self.forward_single(t, self.pos_embed_t)
        return x, t
