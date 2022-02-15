import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import trunc_normal_

#adapted from open-mmlab implementation of swin transformer
class RelativePositionBias(nn.Module):
    def __init__(self,
            window_size=(7, 7),
            num_heads=8
        ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        
        # define parameter table and idx of relative position bias
        Wh, Ww = self.window_size
        num_rows = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rows, num_heads)
        )
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous().view(-1)
        self.register_buffer('relative_position_index', rel_position_index)
        self.init_weights()
    
    def init_weights(self): #important!
        trunc_normal_(self.relative_position_bias_table, std=0.02)
  
    def forward(self, *args, **kwargs):
        Wh, Ww = self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index]
        bias = bias.view(Wh * Ww, Wh * Ww, -1) 
        bias = bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        bias = bias.unsqueeze(0) # 1 nH Wh*Ww Wh*Ww
        return bias
    
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)
