import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, ATTENTION
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..position import RelativePositionBias
from ..utils.reshape import Windower
from ..builder import BLOCKS

@BLOCKS.register_module()
class WindowAttentionBlock(BaseModule):
    def __init__(self,
            window_size=(7, 7),
            shift_size=(0, 0),
            attn_cfg=dict(),
            norm_cfg=dict(type='LN'),
            dropout_cfg=dict(type='DropPath', drop_prob=0.1),
            init_cfg=None
        ):
        super(WindowAttentionBlock, self).__init__(init_cfg)
        self.windower = Windower(window_size, shift_size)
        self.attn = build_from_cfg(attn_cfg, ATTENTION)
        self.norm = build_norm_layer(norm_cfg, self.attn.v_dim)[1]
        self.dropout = build_from_cfg(dropout_cfg, DROPOUT_LAYERS)
        self.attn_bias = RelativePositionBias(window_size, self.attn.num_heads)
    
    def forward(self, x, t=None):
        B, H, W, C = x.shape
        x = self.windower(x) #B nW Ww Wh C
        B, nW, Wh, Ww, C = x.shape
        x = x.view(B*nW, Wh*Ww, C) #B L C
                
        identity = x
        x = self.norm(x)
        x = self.attn(x, x, x, offset=self.attn_bias())
        x = self.dropout(x)
        x = x + identity
        
        x = x.view(B, nW, Wh, Ww, C)
        x = self.windower.undo(x, H, W) #B H W C
        return x, t
