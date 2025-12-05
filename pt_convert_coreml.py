#!/usr/bin/env python3
# pt_convert_coreml.py - Convert AlphaPose PyTorch model to CoreML
import torch
import coremltools as ct
from pathlib import Path
import torch.nn as nn

# AlphaPose Fast Pose model architecture
# Based on FastPose (Fast50) with regression head


class FastPose(nn.Module):
    """
    AlphaPose FastPose model for single-person pose estimation.
    Uses ResNet-50 backbone with regression head for direct coordinate prediction.
    """

    def __init__(self, num_joints=17):
        super(FastPose, self).__init__()
        # Import here to avoid unnecessary dependency if model is already traced
        import torchvision.models as models

        # Use ResNet-50 as backbone
        resnet = models.resnet50(pretrained=False)

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Regression head for direct coordinate prediction
        # Input: 2048 channels from ResNet-50
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 17 joints * 2 coordinates (x, y)
        self.fc = nn.Linear(2048, num_joints * 2)

    def forward(self, x):
        # x shape: (batch, 3, height, width)
        features = self.backbone(x)  # (batch, 2048, h/32, w/32)
        pooled = self.avgpool(features)  # (batch, 2048, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, 2048)
        coords = self.fc(pooled)  # (batch, num_joints * 2)

        # Reshape to (batch, num_joints, 2)
        batch_size = coords.shape[0]
        coords = coords.view(batch_size, -1, 2)

        return coords


def load_model(checkpoint_path, num_joints=17):
    """Load the AlphaPose model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Create model
    model = FastPose(num_joints=num_joints)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print("✓ Model loaded successfully")
    return model


def convert_to_coreml(model, input_shape=(1, 3, 256, 192), output_path="models/alphapose.mlpackage"):
    """Convert PyTorch model to CoreML."""
    print(f"\n{'='*60}")
    print("Converting to CoreML...")
    print('='*60)

    # Create example input
    example_input = torch.rand(*input_shape)

    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)
    print("✓ Model traced successfully")

    # Convert to CoreML
    print("Converting to CoreML format...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="image",
                shape=ct.Shape(shape=input_shape),
            )
        ],
        outputs=[
            ct.TensorType(
                name="coordinates",
                # Output shape: (1, 17, 2) for 17 joints with (x, y) coordinates
            )
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,  # FP16 for Neural Engine
        compute_units=ct.ComputeUnit.ALL,  # Enable Neural Engine + GPU + CPU
        minimum_deployment_target=ct.target.macOS13,
    )

    # Add model metadata
    mlmodel.short_description = "AlphaPose Fast50 pose estimation model"
    mlmodel.author = "AlphaPose"
    mlmodel.license = "GPL-3.0"

    # Add input/output descriptions
    mlmodel.input_description["image"] = "Input image tensor (1, 3, 256, 192) - RGB format"
    mlmodel.output_description[
        "coordinates"] = "Predicted joint coordinates (1, 17, 2) - normalized (x, y) for 17 COCO keypoints"

    # Save the model
    output_path = Path(output_path)
    mlmodel.save(str(output_path))

    print(f"\n{'='*60}")
    print("✓ Conversion successful!")
    print('='*60)
    print(f"Saved: {output_path}")
    print(f"\nModel details:")
    print(f"  Input shape: {input_shape}")
    print(f"  Output: (1, 17, 2) - 17 joints with (x, y) coordinates")
    print(f"  Compute precision: FLOAT16")
    print(f"  Compute units: ALL (Neural Engine + GPU + CPU)")

    print(f"\nNow compile the model:")
    print(f"  cd models")
    print(f"  xcrun coremlcompiler compile alphapose.mlpackage .")

    return mlmodel


def main():
    # Configuration
    checkpoint_path = Path("models/multi_domain_fast50_regression_256x192.pth")
    output_path = Path("models/alphapose.mlpackage")
    input_shape = (1, 3, 256, 192)  # Batch=1, RGB, Height=256, Width=192
    num_joints = 17  # COCO format: 17 keypoints

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Load the PyTorch model
    model = load_model(checkpoint_path, num_joints=num_joints)

    # Convert to CoreML
    mlmodel = convert_to_coreml(
        model, input_shape=input_shape, output_path=str(output_path))

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
