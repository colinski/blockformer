import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#x and y are [bs1,bs2,...,bk] x n
#interleave along feat dim
#output is [bs1,bs2,...,bk] x 2n
def interleave(x, y):
    z = torch.stack([x, y], dim=-1) #bs x n x 2
    z = z.flatten(-2) #bs x 2*n
    return z
#built from mmdetection and CondDETR implementations
#code is very different but computes the same thing
class SineTransform(nn.Module):
    def __init__(self, dim=128, scale=2*math.pi):
        super().__init__()
        assert dim % 2 == 0, 'embedding dim must be even'
        logspace = torch.logspace(start=1, end=-4, steps=dim//2, base=scale)
        self.register_buffer('logspace', logspace) #auto move to gpu/cpu

    #offset is [bs1,bs2,...,bk] x n 
    def forward(self, offset):
        pos = offset.unsqueeze(-1) * self.logspace #bs x dim/2
        pos = interleave(pos.sin(), pos.cos())
        return pos.squeeze(-1)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, shape=(100, 256)):
        super().__init__()
        self.shape = shape
        self.embedding = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        return x + self.embedding

class SinePositionalEncoding(nn.Module):
    def __init__(self, dim=256, scale=2 * math.pi):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.sine_transform = SineTransform(dim//2, scale=scale)

    def forward(self, mask):
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y = not_mask.cumsum(1, dtype=torch.float32)
        x = not_mask.cumsum(2, dtype=torch.float32)
        pos_y = self.sine_transform(y / y.max())
        pos_x = self.sine_transform(x / x.max())
        pos = torch.cat([pos_y, pos_x], dim=-1)
        return pos
