import torch
from blockformer.blocks.stem import ResnetStemBlock
# from blockformer.blocks.window_attn import WindowAttentionBlock
# from blockformer.blocks.pixel_attn import PixelAttentionBlock
# from blockformer.attention import QKVAttention
# from onnxsim import simplify
# import onnx


stem_checkpoint = 'checkpoints/resnet50_stem.pth'

init_cfg = dict(type='Pretrained', checkpoint=stem_checkpoint)

model = ResnetStemBlock(init_cfg=init_cfg)

model.init_weights()

print(model.stem[0].weight.mean())
sd = torch.load(stem_checkpoint)
print(sd['stem.0.weight'].mean())
