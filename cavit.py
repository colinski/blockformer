# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
#from .cond_attentionv2 import MultiheadAttention
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING, TRANSFORMER_LAYER)
#from mmcv.runner.base_module import BaseModule
from mmcv.utils import build_from_cfg
from mmdet.models.utils.positional_encoding import SineTransform
from mmdet.models.utils.res_ffn import MLP

@TRANSFORMER.register_module()
class CrossAttentionTransformerV2(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 num_layers=[2,2,6], #multi_scale case
                 sa_cfg=None,
                 ca_cfg=None,
                 mlp_cfg=None,
                 dropout=0.1,
                 add_before_ca=True
        ):
        super().__init__()
        blocks = []
        for layer_count in num_layers:
            block = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_layers=layer_count,
                dropout=dropout,
                sa_cfg=sa_cfg,
                ca_cfg=ca_cfg,
                mlp_cfg=mlp_cfg,
                add_before_ca=add_before_ca
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.embed_dim = embed_dim
        self.init_weights()
    
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, tgt_pos, memory_pos, memory_mask): 
        all_out, all_ref_points = [], []
        for i in range(len(self.blocks)):
            hs, ref_points = self.blocks[i](
                tgt, memory[i], 
                tgt_pos, memory_pos[i],
                memory_mask[i]
            )
            all_out.append(hs)
            all_ref_points.append(ref_points.unsqueeze(0).repeat(len(hs),1,1,1))
            tgt = hs[-1].permute(1, 0, 2) #num_targets x bs x embed_dim
        ref_points = torch.cat(all_ref_points, dim=0) 
        hs = torch.cat(all_out, dim=0)
        return hs, ref_points


@TRANSFORMER_LAYER.register_module()
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=256, num_layers=6, dropout=0.1,
            sa_cfg=None,
            ca_cfg=None,
            mlp_cfg=None,
            add_before_ca=True
        ):
        super().__init__()
        layers = [] 
        for i in range(num_layers):
            layer = CrossAttentionLayer(
                embed_dim,
                sa_cfg=sa_cfg,
                ca_cfg=ca_cfg,
                mlp_cfg=mlp_cfg,
                add_before_ca=(i==0 and add_before_ca)
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.shared_norm = nn.LayerNorm(embed_dim)
        if num_layers > 1:
            self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(embed_dim, embed_dim, 2, 2)
        self.sine_transform = SineTransform(embed_dim//2)
        self.init_weights()
    
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
       
    #memory: bs x nc x h x w
    #memory_pos: bs x nc x h x w
    #memory_mask: bs x h x w
    def forward(self, memory, tgt,
                tgt_pos=None, memory_pos=None,
                memory_mask=None, tgt_mask=None):
        ref_points = self.ref_point_head(tgt_pos).sigmoid() #ntargets x bs x 2
        # ref_points = self.ref_point_head(tgt_pos).sigmoid().transpose(0, 1)
        # obj_center = ref_points[..., :2].transpose(0, 1)      # [num_queries, batch_size, 2]
        ca_tgt_pos = torch.cat([
            self.sine_transform(ref_points[..., 1]), #1 before 0 matches orig function
            self.sine_transform(ref_points[..., 0])],
            dim=-1
        )
        
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            pos_scale = self.query_scale(tgt) if layer_id != 0 else 1
            tgt = layer(
                tgt, memory, 
                sa_tgt_pos=tgt_pos,
                ca_tgt_pos=(ca_tgt_pos * pos_scale),
                memory_pos=memory_pos, 
                memory_mask=memory_mask
            )
            tgt = self.shared_norm(tgt)
            intermediate.append(tgt)

        # tgt = self.shared_norm(tgt)
        # intermediate.pop()
        # intermediate.append(tgt)
        hs = torch.stack(intermediate)
        return hs.transpose(1, 2), ref_points.transpose(0, 1)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#DOES CONDDetr use in_proj and out_proj on attn?
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, 
                 dropout=0.1,
                 sa_cfg=dict(type='QKVAttention',
                     qk_dim=256,
                     num_heads=8,
                     attn_drop=0.1
                ),
                ca_cfg=dict(type='QKVAttention',
                    embed_dim=256*2,
                    vdim=256,
                    nhead=8,
                    attn_drop=0.1
                ),
                mlp_cfg=dict(type='MLP',
                    input_dim=256,
                    hidden_dim=2048,
                    output_dim=256,
                    num_layers=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    dropout=0.1
                ),
                add_before_ca=False
        ):
        super().__init__()
        # self attention
        self.sa_qcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_qpos_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_kcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_kpos_proj = nn.Linear(embed_dim, embed_dim)
        self.sa_v_proj = nn.Linear(embed_dim, embed_dim)
        self.self_attn = build_from_cfg(sa_cfg, ATTENTION)

        # cross attention
        self.ca_qcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.ca_qpos_proj = nn.Linear(embed_dim, embed_dim)
        self.ca_kcontent_proj = nn.Linear(embed_dim, embed_dim)
        self.ca_kpos_proj = nn.Linear(embed_dim, embed_dim)
        self.ca_v_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = build_from_cfg(ca_cfg, ATTENTION)
        
        if add_before_ca:
            self.add_before_proj = nn.Linear(embed_dim, embed_dim)

        # everything else
        self.mlp = build_from_cfg(mlp_cfg, FEEDFORWARD_NETWORK)
        self.sa_norm = nn.LayerNorm(embed_dim)
        self.ca_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()
        self.add_before_ca = add_before_ca
    
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                memory, #hw x bs x nc
                tgt, #num_tgt x bs x nc
                memory_pos=None,
                memory_mask=None
                sa_tgt_pos=None,
                ca_tgt_pos=None,
        ):
        # ========== Begin of Self-Attention =============
        q = self.sa_qcontent_proj(tgt) + self.sa_qpos_proj(sa_tgt_pos)
        k = self.sa_kcontent_proj(tgt) + self.sa_kpos_proj(sa_tgt_pos)
        v = self.sa_v_proj(tgt)

        h = self.self_attn(query=q, key=k, value=v)[0]
        tgt = tgt + self.dropout(h)
        tgt = self.sa_norm(tgt)
        # ========== End of Self-Attention =============


        # ========== Begin of Cross-Attention =============
        q = self.ca_qcontent_proj(tgt)
        k = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        kpos = self.ca_kpos_proj(memory_pos)
        
        #reshape q and k
        num_targets, bs, embed_dim = q.shape
        hw, _, _ = k.shape
        nhead = self.cross_attn.nhead
        head_dim = embed_dim // nhead
        if self.add_before_ca:
            qpos = self.add_before_proj(sa_tgt_pos)
            q = q + qpos
            k = k + kpos

        q = q.view(num_targets, bs, nhead, head_dim)
        qpos = self.ca_qpos_proj(ca_tgt_pos).view(q.shape)
        q = torch.cat([q, qpos], dim=3).view(num_targets, bs, embed_dim * 2)
        
        k = k.view(hw, bs, nhead, head_dim)
        kpos = kpos.view(k.shape)
        k = torch.cat([k, kpos], dim=3).view(hw, bs, embed_dim * 2)

        h = self.cross_attn(query=q, key=k, value=v, key_padding_mask=memory_mask)[0]               
        tgt = tgt + self.dropout(h)
        tgt = self.ca_norm(tgt)
        # ========== End of Cross-Attention =============

        h = self.mlp(tgt)
        tgt = tgt + self.dropout(h)
        tgt = self.mlp_norm(tgt)
        return memory, tgt
