#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct

from mmpose.apis import init_model
from mmpose.utils import register_all_modules


class RTMPoseCoreMLWrapper(nn.Module):
    """
    Wrap MMPose RTMPose model so CoreML sees:

      input:  (N, 3, H, W) float tensor
      output: pure tensor(s) from mode="tensor"
    """

    def __init__(self, pose_model: nn.Module):
        super().__init__()
        self.pose_model = pose_model

    def forward(self, x: torch.Tensor):
        # Most RTMPose configs support this export-style forward
        out = self.pose_model(x, data_samples=None, mode="tensor")
        return out


def convert_to_coreml(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    image_height: int,
    image_width: int,
):
    print("=" * 60)
    print("Registering MMPose modules...")
    register_all_modules()

    print("Initializing RTMPose model...")
    model = init_model(config_path, checkpoint_path, device="cpu")
    model.eval()
    print("Base model:", type(model))

    wrapped = RTMPoseCoreMLWrapper(model).eval()

    example_input = torch.rand(
        1, 3, image_height, image_width, dtype=torch.float32)
    print("\nTracing with input shape:", tuple(example_input.shape))

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input)
        traced = torch.jit.freeze(traced)

        out = traced(example_input)
        if isinstance(out, torch.Tensor):
            print("Output tensor shape:", tuple(out.shape))
        elif isinstance(out, (list, tuple)):
            print("Output sequence shapes:", [tuple(t.shape) for t in out])
        else:
            print("Output type:", type(out))

    print("\nConverting to CoreML (ML Program)...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input",
                shape=ct.Shape(example_input.shape),
            )
        ],
        # On Linux we *only* convert; you’ll run on macOS/iOS later
        compute_precision=ct.precision.FLOAT16,  # FP16 for Neural Engine
        compute_units=ct.ComputeUnit.ALL,  # Enable Neural Engine + GPU + CPU
        minimum_deployment_target=ct.target.macOS13,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))

    print("\n" + "=" * 60)
    print("✓ Conversion successful!")
    print("=" * 60)
    print("Saved:", out_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert RTMPose (MMPose) .pth to CoreML .mlpackage"
    )
    p.add_argument("--config", required=True, help="RTMPose MMPose config .py")
    p.add_argument("--checkpoint", required=True,
                   help="RTMPose checkpoint .pth")
    p.add_argument("--output", required=True, help="Output .mlpackage path")
    p.add_argument("--height", type=int, required=True,
                   help="Input image height")
    p.add_argument("--width", type=int, required=True,
                   help="Input image width")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_coreml(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_height=args.height,
        image_width=args.width,
    )
