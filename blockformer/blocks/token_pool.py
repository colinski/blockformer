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
class TokenPoolBlock(BaseModule):
    def __init__(self,init_cfg=None):
        super(TokenPoolBlock, self).__init__(init_cfg)

    def forward(self, x, t):
        t = t.mean(dim=1) #B C
        return x, t
