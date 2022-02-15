from mmcv import build_from_cfg
from mmcv.utils import Registry

BLOCKS = Registry('block')

def build_block(cfg):
    return build_from_cfg(cfg, BLOCKS)
