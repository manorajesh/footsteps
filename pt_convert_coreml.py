import coremltools as ct
import torch
from alphapose.models.builder import build_sppe
from alphapose.utils.config import update_config
from pathlib import Path

# 1) Load config
cfg = update_config('models/configs/256x192_res50_lr1e-3_2x-regression.yaml')

# 2) Build model
pose_model = build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

# 3) Load weights
state = torch.load(
    'models/multi_domain_fast50_regression_256x192.pth', map_location='cpu')
pose_model.load_state_dict(state, strict=True)

pose_model.eval()
print('Model ready:', type(pose_model))

# Trace the model with random data.
example_input = torch.rand(1, 3, 256, 192)
traced_model = torch.jit.trace(pose_model, example_input)

print(f"\n{'='*60}")
print("Converting to CoreML...")
print('='*60)

# Convert to Core ML program using the Unified Conversion API.
model_from_trace = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(
            name="input",
            shape=ct.Shape(shape=example_input.shape),
        )
    ],
    compute_precision=ct.precision.FLOAT16,  # FP16 for Neural Engine
    compute_units=ct.ComputeUnit.ALL,  # Enable Neural Engine + GPU + CPU
    minimum_deployment_target=ct.target.macOS13,
)

output_path = Path("models/alphapose.mlpackage")
model_from_trace.save(str(output_path))

print(f"\n{'='*60}")
print("✓ Conversion successful!")
print('='*60)
print(f"Saved: {output_path}")
print(f"\nNow compile the model:")
print(f"  cd models")
print(f"  xcrun coremlcompiler compile alphapose.mlpackage .")
print(f"\nThen run the app:")
print(f"  cargo run --release -- models/alphapose.mlmodelc")
