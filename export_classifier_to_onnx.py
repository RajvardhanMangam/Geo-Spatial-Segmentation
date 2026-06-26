import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(name, num_classes):
    name = name.lower()
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if name == "convnext_small":
        model = models.convnext_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError("Unknown model. Use convnext_tiny, convnext_small, efficientnet_v2_s")


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt
    raise ValueError(
        "Expected a training.py checkpoint dictionary containing keys like "
        "'model', 'classes', 'model_name', and 'image_size'."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-dynamic-batch", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint_path)
    classes = ckpt["classes"]
    model_name = ckpt["model_name"]
    image_size = int(ckpt["image_size"])

    model = build_model(model_name, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    dynamic_axes = None
    if not args.no_dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    metadata_path = output_path.with_suffix(".classes.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": classes,
                "model_name": model_name,
                "image_size": image_size,
                "input": "float32 NCHW RGB, normalized with ImageNet mean/std",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            f,
            indent=2,
        )

    print(f"Saved ONNX model: {output_path}")
    print(f"Saved class metadata: {metadata_path}")


if __name__ == "__main__":
    main()
