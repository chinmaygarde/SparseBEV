import torch
import torch.nn.functional as F

try:
    from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c2345_forward, _ms_deform_attn_cuda_c2345_backward
    from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c23456_forward, _ms_deform_attn_cuda_c23456_backward
    MSMV_CUDA = True
except ImportError as e:
    print('Warning: failed to load one or more CUDA extensions, performance may be hurt.')
    print('Error message:', e)
    MSMV_CUDA = False


def msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights):
    """
    value: [B, N, H1W1 + H2W2..., C]
    sampling_locations: [B, Q, P, 3]
    scale_weights: [B, Q, P, 4]
    """
    assert scale_weights.shape[-1] == len(mlvl_feats)

    B, C, _, _, _ = mlvl_feats[0].shape
    _, Q, P, _ = sampling_locations.shape

    sampling_locations = sampling_locations * 2 - 1
    sampling_locations = sampling_locations[:, :, :, None, :]  # [B, Q, P, 1, 3]

    final = torch.zeros([B, C, Q, P], device=mlvl_feats[0].device)

    for lvl, feat in enumerate(mlvl_feats):
        out = F.grid_sample(
            feat, sampling_locations, mode='bilinear',
            padding_mode='zeros', align_corners=True,
        )[..., 0]  # [B, C, Q, P]
        out = out * scale_weights[..., lvl].reshape(B, 1, Q, P)
        final += out

    return final.permute(0, 2, 1, 3)


class MSMVSamplingC2345(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c2345_forward)
        return _ms_deform_attn_cuda_c2345_forward(
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c2345_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c2345_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight


class MSMVSamplingC23456(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights):
        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights)
        
        assert callable(_ms_deform_attn_cuda_c23456_forward)
        return _ms_deform_attn_cuda_c23456_forward(
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights = ctx.saved_tensors

        assert callable(_ms_deform_attn_cuda_c23456_backward)
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c23456_backward(grad_output.contiguous(), 
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights
        )
        
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight


def msmv_sampling(mlvl_feats, sampling_locations, scale_weights):
    if len(mlvl_feats) == 4 and MSMV_CUDA:
        return MSMVSamplingC2345.apply(*mlvl_feats, sampling_locations, scale_weights)
    elif len(mlvl_feats) == 5 and MSMV_CUDA:
        return MSMVSamplingC23456.apply(*mlvl_feats, sampling_locations, scale_weights)
    else:
        return msmv_sampling_pytorch(mlvl_feats, sampling_locations, scale_weights)


def msmv_sampling_onnx(mlvl_feats, uv, view_idx, scale_weights):
    """
    ONNX-compatible multi-scale multi-view sampling using 4D F.grid_sample.

    Replaces the 5D volumetric grid_sample used in msmv_sampling_pytorch with
    separate per-view 4D grid_samples followed by a torch.gather for view
    selection. All ops are in ONNX opset 16.

    Args:
        mlvl_feats:   list of [BTG, C, N, H, W] channel-first feature maps
        uv:           [BTG, Q, P, 2]  normalised (u, v) in [0, 1]
        view_idx:     [BTG, Q, P]     integer camera-view indices
        scale_weights:[BTG, Q, P, L]  softmax weights over pyramid levels
    Returns:
        [BTG, Q, C, P]
    """
    BTG, C, N, _, _ = mlvl_feats[0].shape
    _, Q, P, _ = uv.shape

    # Convert UV from [0, 1] to [-1, 1] for F.grid_sample
    uv_gs = uv * 2.0 - 1.0  # [BTG, Q, P, 2]

    # Tile UV for all N views: [BTG*N, Q, P, 2]
    # Use expand+contiguous+reshape (maps to ONNX Expand, better CoreML EP support
    # than repeat_interleave which maps to ONNX Tile and can trip up CoreML)
    uv_gs = uv_gs.unsqueeze(1).expand(BTG, N, Q, P, 2).contiguous().reshape(BTG * N, Q, P, 2)

    # Pre-expand view_idx for gathering along the N dim: [BTG, C, 1, Q, P]
    view_idx_g = view_idx[:, None, None, :, :].expand(BTG, C, 1, Q, P)

    final = torch.zeros(BTG, C, Q, P, device=mlvl_feats[0].device, dtype=mlvl_feats[0].dtype)

    for lvl, feat in enumerate(mlvl_feats):
        _, _, _, H_lvl, W_lvl = feat.shape

        # [BTG, C, N, H, W] -> [BTG, N, C, H, W] -> [BTG*N, C, H, W]
        feat_4d = feat.permute(0, 2, 1, 3, 4).reshape(BTG * N, C, H_lvl, W_lvl)

        # 4D grid_sample: [BTG*N, C, Q, P]
        sampled = F.grid_sample(feat_4d, uv_gs, mode='bilinear', padding_mode='zeros', align_corners=True)

        # [BTG*N, C, Q, P] -> [BTG, N, C, Q, P] -> [BTG, C, N, Q, P]
        sampled = sampled.reshape(BTG, N, C, Q, P).permute(0, 2, 1, 3, 4)

        # Gather the selected camera view: [BTG, C, 1, Q, P] -> [BTG, C, Q, P]
        sampled = torch.gather(sampled, 2, view_idx_g).squeeze(2)

        # Accumulate with per-level scale weight
        w = scale_weights[..., lvl].reshape(BTG, 1, Q, P)
        final = final + sampled * w

    return final.permute(0, 2, 1, 3)  # [BTG, Q, C, P]
