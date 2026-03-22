python := "uv run python"

config   := "configs/r50_nuimg_704x256_400q_36ep.py"
weights  := "checkpoints/r50_nuimg_704x256_400q_36ep.pth"
out_dir  := "exports"

# Export the model to ONNX (output goes to exports/ with a descriptive name)
onnx_export config=config weights=weights out_dir=out_dir:
    {{ python }} export_onnx.py \
        --config  {{ config }} \
        --weights {{ weights }} \
        --out-dir {{ out_dir }}

# Export and validate against PyTorch + CoreML EP
onnx_export_validate config=config weights=weights out_dir=out_dir:
    {{ python }} export_onnx.py \
        --config  {{ config }} \
        --weights {{ weights }} \
        --out-dir {{ out_dir }} \
        --validate
