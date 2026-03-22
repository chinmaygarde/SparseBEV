"""
Export SparseBEV to ONNX for inference via ONNX Runtime CoreML EP.

Usage:
    python export_onnx.py \
        --config configs/r50_nuimg_704x256_400q_36ep.py \
        --weights checkpoints/r50_nuimg_704x256_400q_36ep.pth \
        --out sparsebev.onnx

Then run with CoreML EP:
    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession('sparsebev.onnx',
                                providers=['CoreMLExecutionProvider',
                                           'CPUExecutionProvider'])
    outputs = sess.run(None, {'img': img_np, 'lidar2img': l2i_np, 'time_diff': td_np})
    cls_scores, bbox_preds = outputs  # raw logits, apply NMSFreeCoder.decode() separately

Input format (all float32 numpy arrays):
    img        [1, 48, 3, 256, 704]  BGR, pixel values in [0, 255]
    lidar2img  [1, 48, 4, 4]         LiDAR-to-image projection matrices
    time_diff  [1, 8]                seconds since frame-0, one value per frame
                                     (frame 0 = 0.0, frame k = timestamp[0] - timestamp[k])
"""

import argparse
import sys
from unittest.mock import MagicMock

# mmcv is installed without compiled C++ ops (no mmcv-full on macOS).
# SparseBEV doesn't use any mmcv ops at inference time, so stub out the
# missing extension module before anything else imports mmcv.ops.
sys.modules['mmcv._ext'] = MagicMock()

import torch
import numpy as np

# Register all custom mmdet3d modules by importing the local package
sys.path.insert(0, '.')
import models  # noqa: F401  triggers __init__.py which registers DETECTORS etc.

from mmcv import Config
from mmdet3d.models import build_detector
from mmcv.runner import load_checkpoint
from models.onnx_wrapper import SparseBEVOnnxWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='configs/r50_nuimg_704x256_400q_36ep.py')
    parser.add_argument('--weights',  default='checkpoints/r50_nuimg_704x256_400q_36ep.pth')
    parser.add_argument('--out-dir',  default='exports',
                        help='Directory to write the ONNX model into')
    parser.add_argument('--out',      default=None,
                        help='Override output filename (default: derived from config + opset)')
    parser.add_argument('--opset',   type=int, default=18,
                        help='ONNX opset version (18 recommended for torch 2.x)')
    parser.add_argument('--validate', action='store_true',
                        help='Run ORT inference and compare to PyTorch output')
    return parser.parse_args()


def build_dummy_inputs(num_frames=8, num_cameras=6, H=256, W=704):
    """Return (img, lidar2img, time_diff) dummy tensors for export / validation."""
    img       = torch.zeros(1, num_frames * num_cameras, 3, H, W)
    lidar2img = torch.eye(4).reshape(1, 1, 4, 4).expand(1, num_frames * num_cameras, 4, 4).contiguous()
    time_diff = torch.zeros(1, num_frames)
    return img, lidar2img, time_diff


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Resolve output path
    # ------------------------------------------------------------------ #
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    if args.out is None:
        # Derive a descriptive name from the config stem.
        # e.g. configs/r50_nuimg_704x256_400q_36ep.py
        #   -> sparsebev_r50_nuimg_704x256_400q_36ep_opset18.onnx
        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        args.out = os.path.join(args.out_dir,
                                f'sparsebev_{config_stem}_opset{args.opset}.onnx')
    else:
        args.out = os.path.join(args.out_dir, os.path.basename(args.out))

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.weights, map_location='cpu')
    model.eval()

    wrapper = SparseBEVOnnxWrapper(model).eval()

    # ------------------------------------------------------------------ #
    # Dummy inputs
    # ------------------------------------------------------------------ #
    img, lidar2img, time_diff = build_dummy_inputs()

    # ------------------------------------------------------------------ #
    # Reference PyTorch forward (for later numerical comparison)
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        ref_cls, ref_bbox = wrapper(img, lidar2img, time_diff)
    print(f'PyTorch output shapes: cls={tuple(ref_cls.shape)}  bbox={tuple(ref_bbox.shape)}')

    # ------------------------------------------------------------------ #
    # ONNX export
    # ------------------------------------------------------------------ #
    print(f'Exporting to {args.out} (opset {args.opset}) …')
    torch.onnx.export(
        wrapper,
        (img, lidar2img, time_diff),
        args.out,
        opset_version=args.opset,
        input_names=['img', 'lidar2img', 'time_diff'],
        output_names=['cls_scores', 'bbox_preds'],
        do_constant_folding=True,
        verbose=False,
    )
    print('Export done.')

    # ------------------------------------------------------------------ #
    # ONNX model check
    # ------------------------------------------------------------------ #
    import onnx
    model_proto = onnx.load(args.out)
    onnx.checker.check_model(model_proto)
    print('ONNX checker passed.')

    # ------------------------------------------------------------------ #
    # Optional: validate ORT CPU output against PyTorch
    # ------------------------------------------------------------------ #
    if args.validate:
        import onnxruntime as ort

        print('Running ORT CPU validation …')
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        feeds = {
            'img':       img.numpy(),
            'lidar2img': lidar2img.numpy(),
            'time_diff': time_diff.numpy(),
        }
        ort_cls, ort_bbox = sess.run(None, feeds)

        cls_diff  = np.abs(ref_cls.numpy()  - ort_cls).max()
        bbox_diff = np.abs(ref_bbox.numpy() - ort_bbox).max()
        print(f'Max absolute diff — cls: {cls_diff:.6f}   bbox: {bbox_diff:.6f}')

        if cls_diff < 5e-2 and bbox_diff < 5e-2:
            print('Validation PASSED.')
        else:
            print('WARNING: diff is larger than expected — check for unsupported ops.')

        # ------------------------------------------------------------------ #
        # CoreML EP — must pass MLComputeUnits explicitly; without it ORT
        # discards the EP entirely on first partition error instead of falling
        # back per-node to the CPU provider.
        # ------------------------------------------------------------------ #
        print('\nRunning CoreML EP …')
        sess_cml = ort.InferenceSession(
            args.out,
            providers=[
                ('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}),
                'CPUExecutionProvider',
            ],
        )
        cml_cls, cml_bbox = sess_cml.run(None, feeds)
        cml_cls_diff  = np.abs(ref_cls.numpy()  - cml_cls).max()
        cml_bbox_diff = np.abs(ref_bbox.numpy() - cml_bbox).max()
        print(f'CoreML EP max diff — cls: {cml_cls_diff:.6f}   bbox: {cml_bbox_diff:.6f}')


if __name__ == '__main__':
    main()
