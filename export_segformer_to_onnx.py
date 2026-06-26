import argparse
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


class SegFormerOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def load_state_dict(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to segformer .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--base-model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--num-labels", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.base_model,
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(load_state_dict(args.checkpoint))
    model.eval()

    wrapper = SegFormerOnnxWrapper(model).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Saved ONNX model: {output_path}")
    print("Output shape is logits: [batch, num_labels, h/4, w/4].")


if __name__ == "__main__":
    main()
