#!/usr/bin/env python3
# convert_movenet.py
import tensorflow as tf
import coremltools as ct
from pathlib import Path

# Path to the SavedModel directory
SAVED_MODEL_DIR = Path("models/movenet-tensorflow2-singlepose-thunder-v4")

print(f"Loading SavedModel from: {SAVED_MODEL_DIR}")

# Load to inspect the signature
model = tf.saved_model.load(str(SAVED_MODEL_DIR))
concrete_fn = model.signatures["serving_default"]

print("\nInput signature:", concrete_fn.structured_input_signature)
print("Outputs:", concrete_fn.structured_outputs)

# Get input details
_, input_spec = concrete_fn.structured_input_signature
(input_name, input_tensor_spec), = input_spec.items()
print(
    f"\nUsing input: name={input_name}, dtype={input_tensor_spec.dtype}, shape={input_tensor_spec.shape}")

# Choose a fixed spatial size (must be multiple of 32 for MoveNet)
H, W = 256, 256

print(f"\n{'='*60}")
print("Converting to CoreML...")
print('='*60)

# Convert using the SavedModel path directly (not concrete_fn)
mlmodel = ct.convert(
    str(SAVED_MODEL_DIR),  # Pass the directory path, not the concrete function
    source="tensorflow",
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(
            name=input_name,
            shape=ct.Shape(shape=(1, H, W, 3)),  # Use ct.Shape for clarity
        )
    ],
    compute_precision=ct.precision.FLOAT16,  # FP16 for Neural Engine
    compute_units=ct.ComputeUnit.ALL,  # Enable Neural Engine + GPU + CPU
    minimum_deployment_target=ct.target.macOS13,
)

# Save to models directory
output_path = Path("models/movenet_singlepose_thunder.mlpackage")
mlmodel.save(str(output_path))

print(f"\n{'='*60}")
print("✓ Conversion successful!")
print('='*60)
print(f"Saved: {output_path}")
print(f"\nNow compile the model:")
print(f"  cd models")
print(f"  xcrun coremlcompiler compile movenet_singlepose_thunder.mlpackage .")
print(f"\nThen run the app:")
print(f"  cargo run --release -- models/movenet_singlepose_thunder.mlmodelc")
