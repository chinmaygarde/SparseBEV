import torch
import torch.nn as nn


class SparseBEVOnnxWrapper(nn.Module):
    """
    Thin wrapper around SparseBEV for ONNX export.

    Accepts pre-computed tensors instead of the img_metas dict so the graph
    boundary is clean.  Returns raw decoder logits without NMS or decoding so
    post-processing can stay in Python.

    Inputs (all float32):
        img        [B, T*N, 3, H, W]  — BGR images, will be normalised inside
        lidar2img  [B, T*N, 4, 4]     — LiDAR-to-image projection matrices
        time_diff  [B, T]             — seconds since the first frame (per frame,
                                        averaged across the N cameras)

    Outputs:
        cls_scores  [num_layers, B, Q, num_classes]
        bbox_preds  [num_layers, B, Q, 10]
    """

    def __init__(self, model, image_h=256, image_w=704, num_frames=8, num_cameras=6):
        super().__init__()
        self.model = model
        self.image_h = image_h
        self.image_w = image_w
        self.num_frames = num_frames
        self.num_cameras = num_cameras

        # Disable stochastic augmentations that are meaningless at inference
        self.model.use_grid_mask = False
        # Disable FP16 casting decorators
        self.model.fp16_enabled = False

    def forward(self, img, lidar2img, time_diff):
        B, TN, C, H, W = img.shape

        # Build a minimal img_metas.  Only the Python-constant fields are here;
        # the tensor fields (time_diff, lidar2img) are injected as real tensors
        # so the ONNX tracer includes them in the graph.
        img_shape = (self.image_h, self.image_w, C)
        img_metas = [{
            'img_shape': [img_shape] * TN,
            'ori_shape': [img_shape] * TN,
            'time_diff': time_diff,    # tensor — flows into the ONNX graph
            'lidar2img': lidar2img,    # tensor — flows into the ONNX graph
        }]

        # Backbone + FPN
        img_feats = self.model.extract_feat(img=img, img_metas=img_metas)

        # Detection head — returns raw predictions, no NMS
        outs = self.model.pts_bbox_head(img_feats, img_metas)

        cls_scores = outs['all_cls_scores']   # [num_layers, B, Q, num_classes]
        bbox_preds = outs['all_bbox_preds']   # [num_layers, B, Q, 10]

        return cls_scores, bbox_preds
