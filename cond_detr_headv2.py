# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

#from cond detr repo
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

@HEADS.register_module()
class CondDETRHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss=dict(type='SetBasedLoss',
                     num_classes=80,
                     cost_class=2.0,
                     cost_bbox=2.0,
                     cost_giou=5.0,
                     focal_alpha=0.25,
                     focal_gamma=2.0
                 ),
                 # loss_bbox=dict(type='L1Loss', loss_weight=5.0, reduction='none'),
                 # loss_iou=dict(type='GIoULoss', loss_weight=2.0, reduction='none'),
                 # train_cfg=dict(
                     # assigner=dict(
                         # type='HungarianAssigner',
                         # cls_cost=dict(type='ClassificationCost', weight=1.),
                         # reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         # iou_cost=dict(
                             # type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=300),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        #self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_fn = build_loss(loss)
        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        # self.loss_iou = build_loss(loss_iou)

        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        # assert 'num_feats' in positional_encoding
        # num_feats = positional_encoding['num_feats']
        # assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            # f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            # f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.fc_cls = nn.Linear(self.embed_dims, self.num_classes)
        self.fc_cls.bias.data = torch.ones(self.num_classes) * bias_value

        # init bbox_mebed
        self.reg_ffn = MLP(self.embed_dims, self.embed_dims, 4, 3)
        nn.init.constant_(self.reg_ffn.layers[-1].weight.data, 0)
        nn.init.constant_(self.reg_ffn.layers[-1].bias.data, 0)

        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1)

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    
    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        # if torch.onnx.is_in_onnx_export():
            # return self.forward_single(feats[0], img_metas)
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        # return multi_apply(self.forward_single, feats, img_metas_list)
        return self.forward_single(feats[0], img_metas_list[0])

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        
        if torch.onnx.is_in_onnx_export(): #interpolate breaks onnx->trt
            assert batch_size == 1
            masks = x.new_zeros((batch_size, x.shape[2], x.shape[3]))
        else:
            masks = F.interpolate(
                masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]

        hs, ref_points = self.transformer(x, masks, self.query_embedding.weight, pos_embed)

        reference_before_sigmoid = inverse_sigmoid(ref_points)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.reg_ffn(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = self.fc_cls(hs)
        return outputs_class, outputs_coords
        
    def loss(self, outputs, targets):
        return self.loss_fn(outputs, targets)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self.forward(x, img_metas)
        bs = len(gt_labels)
        
        targets = []
        for i in range(bs):
            #bbox_pred = outs['pred_boxes'][i]
            img_h, img_w, _ = img_metas[i]['img_shape']
            #factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
            factor = gt_bboxes[i].new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)#.repeat(bbox_pred.size(0), 1)
            #outs['pred_boxes'][i] = bbox_pred * factor
            boxes = bbox_xyxy_to_cxcywh(gt_bboxes[i]) / factor
            labels = gt_labels[i]
            targets.append({'labels': labels, 'boxes': boxes})
        # factors = []
        # for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            # img_h, img_w, _ = img_meta['img_shape']
            # factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           # img_h]).unsqueeze(0).repeat(
                                               # bbox_pred.size(0), 1)
            # factors.append(factor)
        # factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        # bbox_preds = bbox_preds.reshape(-1, 4)
        # bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        # bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # targets = []
        # for i in range(bs):
            # labels = gt_labels[i]
            # boxes = bbox_xyxy_to_cxcywh(gt_bboxes[i])
            # targets.append({'labels': labels, 'boxes': boxes})
        return self.loss(outs, targets)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1]#[-1] #bs x ndec x 80
        bbox_preds = all_bbox_preds_list[-1]#[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # out_logits, out_bbox = cls_score.unsqueeze(0), bbox_pred.unsqueeze(0)
        # prob = out_logits.sigmoid()
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]
        # boxes = box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        # img_h, img_w, _ = img_shape
        # scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).to(boxes.device)
        # boxes = boxes * scale_fct[:, None, :]
        # det_bboxes = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1).squeeze(0)
        # det_labels = labels.squeeze(0)

        # exclude background
        # if self.loss_cls.use_sigmoid:
            # cls_score = cls_score.sigmoid()
            # scores, indexes = cls_score.view(-1).topk(max_per_img)
            # det_labels = indexes % self.num_classes
            # bbox_index = indexes // self.num_classes
            # bbox_pred = bbox_pred[bbox_index]
        # else:
        #scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        #scores, det_labels = F.softmax(cls_score, dim=-1).max(-1)
        scores, det_labels = cls_score.sigmoid().max(-1)
        scores, bbox_index = scores.topk(max_per_img)
        bbox_pred = bbox_pred[bbox_index]
        det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def get_targets(self):
        return
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        # version = local_metadata.get('version', None)
        # if (version is None or version < 2) and self.__class__ is DETRHead:
            # convert_dict = {
                # '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                # '.multihead_attn.': '.attentions.1.',
                # '.decoder.norm.': '.decoder.post_norm.'
            # }
            # state_dict_keys = list(state_dict.keys())
            # for k in state_dict_keys:
                # for ori_key, convert_key in convert_dict.items():
                    # if ori_key in k:
                        # convert_key = k.replace(ori_key, convert_key)
                        # state_dict[convert_key] = state_dict[k]
                        # del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
   
    
#from detr family of repos
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
