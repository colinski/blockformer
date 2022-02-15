import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..builder import BLOCKS, ATTENTION
#BACKBONES = Registry('models', parent=MMCV_MODELS)


# @BLOCKS.register_module()
# class TokenPositionalEncodingBlock(BaseModule):
    # def __init__(self, attn_cfg=dict(), norm_cfg=dict(type='LN'),
            # dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            # act_cfg=dict(type='GELU'),
            # init_cfg=None
        # ):
        # super(TokenPositionalEncodingBlock, self).__init__(init_cfg)
        # self.attn = build_from_cfg(attn_cfg, ATTENTION)
        # self.norm = build_norm_layer(norm_cfg, self.attn.v_dim)[1]
        # self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
        # self.embed_dim = self.attn.v_dim
        # self.ref_point_head = nn.Sequential(
            # nn.Linear(self.embed_dim, self.embed_dim),
            # build_from_cfg(act_cfg, ACTIVATION_LAYERS),
            # nn.Linear(self.embed_dim, 2),
            # nn.Sigmoid()
        # )
        # self.sine_transform = SineTransform(dim=self.embed_dim//2)

    # def forward(self, x, t):
        # identity = t
        # t = self.norm(t)
        
        # obj_center = self.ref_point_head(t) #B L 2
        # pos_encodings = torch.cat([
            # self.sine_transform(obj_center[..., 0]),
            # self.sine_transform(obj_center[..., 1])],
            # dim=-1
        # ) #B L D

        # t = self.attn(t, pos_encodings, pos_encodings)
        # t = self.dropout(t)
        # t = t + identity
        # return x, t

# @BLOCKS.register_module()
# class TokenExpansionBlock(BaseModule):
    # def __init__(self, in_tokens=1, out_tokens=100, init_cfg=None):
        # super(TokenExpansionBlock, self).__init__(init_cfg)
        # self.out_tokens = out_tokens
        # self.expansion = nn.Linear(in_tokens, out_tokens)
        
    # def forward(self, x, t):
        # t = t.permute(0, 2, 1) #b c nq
        # t = self.expansion(t)
        # t = t.permute(0, 2, 1) #b nq c
        # return x, t

@BLOCKS.register_module()
class TokenAttentionBlock(BaseModule):
    def __init__(self, 
            attn_cfg=dict(), 
            norm_cfg=dict(type='LN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            init_cfg=None
        ):
        super(TokenAttentionBlock, self).__init__(init_cfg)
        self.attn = build_from_cfg(attn_cfg, ATTENTION)
        self.norm = build_norm_layer(norm_cfg, self.attn.v_dim)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)

    def forward(self, x, t):
        identity = t
        t = self.norm(t)
        t = self.attn(t, t, t)
        t = self.dropout(t)
        t = t + identity
        return x, t

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
