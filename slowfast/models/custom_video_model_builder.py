#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""A More Flexible Video models."""
from .build import MODEL_REGISTRY
from slowfast.models.video_model_builder import MViT
import torch
import torch.nn as nn
from . import head_helper
from .build import MODEL_REGISTRY
from slowfast.models.utils import (
    round_width,
)
from torch.nn import functional as F
@MODEL_REGISTRY.register()
class MViT_Relation(MViT):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        depth = cfg.MVIT.DEPTH
        num_heads = cfg.MVIT.NUM_HEADS
        embed_dim = cfg.MVIT.EMBED_DIM
        temporal_size = cfg.DATA.NUM_FRAMES
        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])

            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            embed_dim = dim_out
        self.head = RelationHead(
            dim_in=[embed_dim],
            num_classes=self.num_classes,
            pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
            resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
            scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            aligned=cfg.DETECTION.ALIGNED,
            cfg = cfg
        )

    def forward(self, inputs, meta):
        x, past_human, future_human, past_contex, future_context = inputs
        x = x[0]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        # assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)
        curent_contex = x.clone()
        relation_feature = (past_human, future_human, curent_contex, past_contex, future_context)
        if self.cls_embed_on:
            x = x[:, 1:]
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])
        x = self.head([x],relation_feature, meta)
        return x
    
class AttentionWithSkipNorm(nn.Module):
    def __init__(self, qdim, kdim=None, vdim=None, num_heads=4, batch_first=True) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(qdim, kdim=kdim, vdim=vdim, num_heads=num_heads,batch_first=batch_first)
        self.norm = nn.LayerNorm(qdim)
    
    def forward(self, q, k, v, attn_mask=None):
        attended = self.norm(q + self.attn(q, k, v, need_weights=False, key_padding_mask=attn_mask)[0])
        return attended
    
class PastCurrentFuture(nn.Module):
    def __init__(self, qdim, kdim=None, vdim=None, num_heads=4, batch_first=True) -> None:
        super().__init__()

        self.current_attn = AttentionWithSkipNorm(qdim, kdim, vdim, num_heads, batch_first)
        
        self.past_attn = AttentionWithSkipNorm(qdim, kdim, vdim, num_heads, batch_first)
        
        self.future_attn = AttentionWithSkipNorm(qdim, kdim, vdim, num_heads, batch_first)
        
        self.linear = nn.Linear(qdim*3, qdim)

    def forward(self, q, kv_past, kv_current, kv_future, mask=None):
        current = self.current_attn(q, kv_current, kv_current, attn_mask=mask[0])
        past = self.past_attn(q, kv_past, kv_past, attn_mask=mask[1])
        future = self.future_attn(q, kv_future, kv_future, attn_mask=mask[2])
        return self.linear(torch.cat([past, current, future], dim=-1))
    
class RelationHead(head_helper.ResNetRoIHead):
    def __init__(self, dim_in, num_classes, pool_size, resolution, scale_factor, dropout_rate=0, act_func="softmax", aligned=True, cfg=None):
        super().__init__(dim_in, num_classes, pool_size, resolution, scale_factor, dropout_rate, act_func, aligned)
        self.cfg = cfg
        query_dim = 768
        self.human_human = PastCurrentFuture(query_dim)
        self.human_context = PastCurrentFuture(query_dim)

        if cfg.COMPUTER.SKELETON:
            self.projection_skeleton = nn.Linear(68, query_dim)
            self.skeleton_skeleton = PastCurrentFuture(query_dim)
            self.skeleton_context = PastCurrentFuture(query_dim)
            self.appear_head = nn.Linear(768, 80)
            self.skeleton_head = nn.Linear(768, 80)

        layers = []
        for i in range(self.cfg.COMPUTER.NUM_RELATION_MODULE):
            layers.append(AttentionWithSkipNorm(query_dim, query_dim, query_dim, 4, batch_first=True))
        self.layers = nn.ModuleList(layers)
        # del self.projection
        # self.projection = nn.Linear(768 * 2, 80)
    def forward(self, x, relation_feature, meta):
        bboxes = meta['boxes']
        # bboxes = meta['boxesObject']
        # for bboxes in [meta['boxes'], meta['boxesObject']]:
        past_human, future_human, curent_contex, past_contex, future_context = relation_feature
        for layer in self.layers:
            curent_contex = layer(curent_contex, curent_contex, curent_contex)
        if self.cfg.COMPUTER.SKELETON:
            current_skeleton = self.projection_skeleton(meta['current_skeleton'])
            # past_skeleton = self.projection_skeleton(meta['past_skeleton'])
            # future_skeleton = self.projection_skeleton(meta['future_skeleton'])
        curent_contex = curent_contex[:, 1:]
        img = [curent_contex.transpose(1, 2).reshape(x[0].shape)]
        # x = curent_contex
        assert (
            len(img) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(img[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        img = torch.cat(pool_out, 1)

        B = relation_feature[0].shape[0]
        # Perform dropout.
        if hasattr(self, "dropout"):
            img = self.dropout(img)
        out = img.view(img.shape[0], -1)
        num_dim = out.size(-1)
        curent_human = out.reshape(B, -1, num_dim)
        curent_human = self.human_human(q=curent_human, kv_past=past_human, kv_current=curent_human, kv_future=future_human, mask=[meta['current_human_mask'], meta['past_humanx_mask'], meta['future_human_mask']])
        curent_human = self.human_context(q=curent_human, kv_past=past_contex, kv_current=curent_contex, kv_future=future_context, mask=[None, meta['past_contex_mask'], meta['future_context_mask']])
        if self.cfg.COMPUTER.SKELETON:
            current_skeleton = self.skeleton_skeleton(q=current_skeleton, kv_past=past_human, kv_current=curent_human, kv_future=future_human, mask=[None, None, None])
            current_skeleton = self.skeleton_context(q=current_skeleton, kv_past=past_contex, kv_current=curent_contex, kv_future=future_context, mask=[None, None, None])
            curent_human_constrastive = self.appear_head(curent_human)[:, 0, :]
            current_skeleton_constrastive = self.skeleton_head(current_skeleton)[:, 0, :]
            xcs = curent_human_constrastive @ current_skeleton_constrastive.T 
            u_norm = torch.norm(curent_human_constrastive, dim=1, keepdim=True)
            v_norm = torch.norm(current_skeleton_constrastive, dim=1, keepdim=True)
            normalized_dot_product = xcs / (u_norm * v_norm.T)
            target = torch.eye(B).to(xcs.device)
            constrastive_loss = F.cross_entropy(normalized_dot_product , target, reduction="mean")
            curent_human = curent_human + current_skeleton

            meta['constrastive_loss'] = 0.3*constrastive_loss
        out = torch.masked_select(curent_human.view(-1, num_dim), ~meta["current_human_mask"].view(-1, 1)).view(-1,num_dim)
        out = self.projection(out)
        out = self.act(out)
        return out


