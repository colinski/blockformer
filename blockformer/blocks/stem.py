import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS, ACTIVATION_LAYERS, NORM_LAYERS
from mmcv.runner.base_module import BaseModule#, ModuleList
from mmcv import build_from_cfg
from ..builder import BLOCKS

@BLOCKS.register_module()
class ResnetStemBlock(BaseModule):
    def __init__(self,init_cfg=None):
        super(ResnetStemBlock, self).__init__(init_cfg)
        self.stem = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.init_weights() #will call super class to init from pth file

    def forward(self, x, t):
        with torch.no_grad():
            x = x.permute(0, 3, 1, 2)
            # if self.stem[0].weight.mean() != -0.0014:
                # print(self.stem[0].weight.mean())
            x = self.stem.eval()(x)
            x = x.permute(0, 2, 3, 1)
        return x, t
