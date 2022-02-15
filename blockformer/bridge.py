import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv import build_from_cfg
# from mmcv.cnn import MODEL as MMCV_MODELS
from mmcls.models.builder import BACKBONES, NECKS
from .builder import build_block

@BACKBONES.register_module()
class Blockbone(BaseModule):
    def __init__(self,
            blocks=[],
            num_tokens=1,
            token_dim=3,
            return_x=False,
            init_cfg=None
        ):
        super(Blockbone, self).__init__(init_cfg)
        self.tokens = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
        self.blocks = ModuleList([build_block(cfg) for cfg in blocks])
        self.return_x = return_x
        self.init_weights()
    
    def init_weights(self):
        trunc_normal_(self.tokens, std=0.02)

    #x is B 3 H W
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        t = self.tokens.expand(B, -1, -1)

        for block in self.blocks:
            x, t = block(x, t)
        
        if self.return_x:
            return (x, t)
        return (t,)

# @NECKS.register_module()
# class BlockFormerNeck(BaseModule):
    # def __init__(self,
            # blocks=[],
            # init_cfg=None
        # ):
        # super(BlockFormerNeck, self).__init__(init_cfg)
        # self.blocks = ModuleList([build_block(cfg) for cfg in blocks])
        # self.init_weights()
    
    # def init_weights(self):
        # pass

    # def forward(self, input):
        # x, t = input 
        # for block in self.blocks:
            # x, t = block(x, t)
        # return (x, t)
