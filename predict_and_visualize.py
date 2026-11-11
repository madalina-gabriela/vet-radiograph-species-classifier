import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from dicom_preprocess import dicom_to_uint8


def dicom_to_uint8_display(dcm_path: Path) -> np.ndarray:
    return dicom_to_uint8(dcm_path)


def build_model(num_classes: int = 2, dropout: float = 0.0) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes))
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False,
    )
    with torch.no_grad():
        model.conv1.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to DICOM image.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_model.pt checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save visual prediction PNG.")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    ckpt_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr_u8 = dicom_to_uint8_display(image_path)
    img = transforms.ToPILImage(mode="L")(arr_u8)

    mean = [float(np.mean([0.485, 0.456, 0.406]))]
    std = [float(np.mean([0.229, 0.224, 0.225]))]
    transform = transforms.Compose(
        [
            transforms.Resize(int(args.img_size * 1.15)),
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    x = transform(img).unsqueeze(0)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    class_mapping = checkpoint.get("class_mapping", {"0": "dog", "1": "cat"})
    dropout = float(checkpoint.get("fc_dropout", 0.3))

    model = build_model(num_classes=2, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = class_mapping.get(str(pred_idx), str(pred_idx))
        confidence = float(probs[pred_idx])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(arr_u8, cmap="gray")
    ax.axis("off")
    ax.set_title(
        f"Prediction: {pred_label.upper()}  |  confidence: {confidence:.2%}\n"
        f"P(dog)={probs[0]:.3f}, P(cat)={probs[1]:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

