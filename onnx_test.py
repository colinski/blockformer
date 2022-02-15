import torch
from blockformer.blocks.window_attn import WindowAttentionBlock
from blockformer.blocks.pixel_attn import PixelAttentionBlock
from blockformer.attention import QKVAttention
from onnxsim import simplify
import onnx



window_attn = WindowAttentionBlock(
        attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8),
        window_size=(7, 7), 
        shift_size=(0, 0)
)

pixel_attn = PixelAttentionBlock(
        attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8),
)


# x = torch.randn(1, 56, 56, 256)
x = torch.randn(1, 256, 56, 56)
t = torch.randn(1, 100, 256)
torch.onnx.export(
    pixel_attn,
    (x, t),
    'pixel_attn.onnx',
    input_names=['x', 't'],
    output_names=['x_out', 't_out'],
    export_params=True,
    keep_initializers_as_inputs=True,
    do_constant_folding=True,
    verbose=True,
    opset_version=11
)

model = onnx.load('pixel_attn.onnx')
model, check = simplify(model)
onnx.save(model, 'pixel_attn.onnx')
print(model)
