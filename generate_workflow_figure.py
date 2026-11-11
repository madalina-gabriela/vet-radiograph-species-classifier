#!/usr/bin/env python3
"""Workflow figure — line-based layout to prevent text overlap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PALETTE = {
    "header": "#1a5f4a",
    "header_sub": "#d4ece4",
    "data": ("#eef8f1", "#2d8a5e"),
    "preproc": ("#f5f0fa", "#7e57a0"),
    "split": [("#fff8e1", "#f9a825"), ("#ffebee", "#e53935"), ("#e3f2fd", "#1e88e5")],
    "model": ("#eef5fc", "#1565c0"),
    "results": ("#eef8f1", "#2d8a5e"),
    "gradcam": ("#f5f0fa", "#7e57a0"),
    "arrow": "#8a9590",
    "text": "#222",
    "muted": "#555",
}

# Figure layout: x in [0,1], y in [0, H] (data coordinates, bottom = 0)
H = 100.0
MX, MW = 6.0, 88.0
CX = MX + MW / 2
TITLE_BAR = 3.2
PAD = 1.0
LINE = 1.55  # vertical space per text line (layout units)


class Layout:
    def __init__(self, ax):
        self.ax = ax
        self.y = H - 2.0  # cursor from top

    def gap(self, g: float = 1.2) -> None:
        self.y -= g

    def header_band(self) -> None:
        y0 = self.y - 5.5
        self.ax.add_patch(
            FancyBboxPatch(
                (4, y0),
                92,
                5.5,
                boxstyle="round,pad=0.02",
                facecolor=PALETTE["header"],
                edgecolor="none",
                clip_on=False,
            )
        )
        self.ax.text(50, y0 + 3.6, "AI Veterinary Radiograph Classification", ha="center", va="center", fontsize=15, fontweight="bold", color="white")
        self.ax.text(
            50,
            y0 + 1.5,
            "Experimental Workflow — ResNet-18 Species Classifier",
            ha="center",
            va="center",
            fontsize=9.5,
            color=PALETTE["header_sub"],
        )
        self.y = y0 - PAD

    def panel(self, title: str, bg: str, accent: str, inner_lines: list[tuple[str, float, str, str]]) -> None:
        """inner_lines: (text, x, ha, style) with style in bold|normal|muted"""
        n = max(1, len(inner_lines))
        height = TITLE_BAR + PAD + n * LINE + PAD
        y0 = self.y - height
        self.ax.add_patch(
            FancyBboxPatch(
                (MX, y0),
                MW,
                height,
                boxstyle="round,pad=0.015",
                facecolor=bg,
                edgecolor=accent,
                linewidth=2,
                clip_on=False,
            )
        )
        self.ax.add_patch(Rectangle((MX, y0 + height - TITLE_BAR), MW, TITLE_BAR, facecolor=accent, edgecolor="none"))
        self.ax.text(MX + 1.2, y0 + height - TITLE_BAR / 2, title, ha="left", va="center", fontsize=10, fontweight="bold", color="white")
        ty = y0 + height - TITLE_BAR - PAD
        for text, x, ha, style in inner_lines:
            ty -= LINE
            kw = dict(ha=ha, va="top", fontsize=8.8, color=PALETTE["muted"])
            if style == "bold":
                kw.update(fontweight="bold", color=PALETTE["text"], fontsize=9.2)
            elif style == "normal":
                kw.update(color=PALETTE["text"], fontsize=8.6)
            self.ax.text(x, ty, text, **kw)
        self.y = y0 - PAD

    def arrow(self) -> None:
        y1 = self.y + PAD * 0.5
        y0 = self.y - 0.3
        self.ax.add_patch(
            FancyArrowPatch(
                (CX, y1),
                (CX, y0),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color=PALETTE["arrow"],
                clip_on=False,
            )
        )
        self.y = y0 - PAD * 0.8

    def split_row(self, caption: str, cards: list[tuple[str, str, str, str, tuple[str, str]]]) -> None:
        self.ax.text(CX, self.y, caption, ha="center", va="top", fontsize=9.5, fontweight="bold", color=PALETTE["text"])
        self.y -= 1.8
        card_h = 6.0
        y0 = self.y - card_h
        gap = 1.5
        w = (MW - 2 * gap) / 3
        for i, (title, img_line, pt_line, pct, colors) in enumerate(cards):
            x = MX + i * (w + gap)
            fc, ec = colors
            self.ax.add_patch(
                FancyBboxPatch(
                    (x, y0),
                    w,
                    card_h,
                    boxstyle="round,pad=0.02",
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=1.8,
                    clip_on=False,
                )
            )
            self.ax.text(x + w / 2, y0 + card_h - 1.0, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=PALETTE["text"])
            self.ax.text(x + w / 2, y0 + card_h - 2.35, img_line, ha="center", va="top", fontsize=8.6, color=PALETTE["text"])
            self.ax.text(x + w / 2, y0 + card_h - 3.55, pt_line, ha="center", va="top", fontsize=8.6, color=PALETTE["muted"])
            self.ax.text(x + w / 2, y0 + 0.75, f"({pct} of images)", ha="center", va="bottom", fontsize=8, color=ec, fontstyle="italic")
        self.y = y0 - PAD

    def two_panels_bottom(self, left_title, left_lines, right_title, right_lines) -> None:
        ph = max(len(left_lines), len(right_lines)) * LINE + TITLE_BAR + PAD * 2 + 2
        y0 = self.y - ph
        hw = (MW - 2) / 2
        for col, (title, lines, colors) in enumerate(
            [
                (left_title, left_lines, PALETTE["results"]),
                (right_title, right_lines, PALETTE["gradcam"]),
            ]
        ):
            x = MX + col * (hw + 2)
            bg, ac = colors
            self.ax.add_patch(
                FancyBboxPatch((x, y0), hw, ph, boxstyle="round,pad=0.015", facecolor=bg, edgecolor=ac, linewidth=2, clip_on=False)
            )
            self.ax.add_patch(Rectangle((x, y0 + ph - TITLE_BAR), hw, TITLE_BAR, facecolor=ac, edgecolor="none"))
            self.ax.text(x + 1.2, y0 + ph - TITLE_BAR / 2, title, ha="left", va="center", fontsize=9.5, fontweight="bold", color="white")
            ty = y0 + ph - TITLE_BAR - PAD
            for text, style in lines:
                ty -= LINE
                if style == "hero":
                    self.ax.text(x + hw / 2, ty, text, ha="center", va="top", fontsize=22, fontweight="bold", color=ac)
                elif style == "bold":
                    self.ax.text(x + hw / 2, ty, text, ha="center", va="top", fontsize=9, fontweight="bold", color=PALETTE["text"])
                else:
                    self.ax.text(x + hw / 2, ty, text, ha="center", va="top", fontsize=8.2, color=PALETTE["muted"])
        self.y = y0 - PAD

    def model_train_row(self) -> None:
        ph = 8.5
        y0 = self.y - ph
        bg, ac = PALETTE["model"]
        self.ax.add_patch(FancyBboxPatch((MX, y0), MW, ph, boxstyle="round,pad=0.015", facecolor=bg, edgecolor=ac, linewidth=2, clip_on=False))
        self.ax.add_patch(Rectangle((MX, y0 + ph - TITLE_BAR), MW, TITLE_BAR, facecolor=ac, edgecolor="none"))
        self.ax.text(MX + 1.2, y0 + ph - TITLE_BAR / 2, "3 · MODEL  &  4 · TRAINING", ha="left", va="center", fontsize=10, fontweight="bold", color="white")
        hw = MW / 2 - 2
        blocks = [
            (
                MX + 1.5,
                [
                    ("ResNet-18 (ImageNet Pretrained)", "bold"),
                    ("Replace final FC → 512-dim feature vector", "normal"),
                    ("→ 2 logits (cat / dog)", "normal"),
                ],
            ),
            (
                MX + MW / 2 + 0.5,
                [
                    ("Training — 10 Epochs", "bold"),
                    ("Adam lr=1e-4 · weight decay=1e-5 · batch=16", "normal"),
                    ("Cross-entropy loss · best checkpoint on val accuracy", "normal"),
                ],
            ),
        ]
        for x, lines in blocks:
            ty = y0 + ph - TITLE_BAR - 1.2
            for text, style in lines:
                ty -= LINE
                kw = dict(ha="left", va="top", fontsize=8.6, color=PALETTE["text"])
                if style == "bold":
                    kw["fontweight"] = "bold"
                    kw["fontsize"] = 9.2
                self.ax.text(x, ty, text, **kw)
        self.y = y0 - PAD


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "test1" / "workflow_diagram_vet_dataset.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 14), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    L = Layout(ax)
    L.header_band()

    bg, ac = PALETTE["data"]
    cw = MW / 3 - 1
    # Data as one panel with 3 columns of fixed lines
    height = TITLE_BAR + PAD + 4 * LINE + PAD
    y0 = L.y - height
    ax.add_patch(FancyBboxPatch((MX, y0), MW, height, boxstyle="round,pad=0.015", facecolor=bg, edgecolor=ac, linewidth=2, clip_on=False))
    ax.add_patch(Rectangle((MX, y0 + height - TITLE_BAR), MW, TITLE_BAR, facecolor=ac, edgecolor="none"))
    ax.text(MX + 1.2, y0 + height - TITLE_BAR / 2, "1 · DATA", ha="left", va="center", fontsize=10, fontweight="bold", color="white")
    cols = [
        ("Veterinary Hospital Dataset", ["592 DICOM radiographs", "366 canine · 226 feline", "187 patients"]),
        ("Inclusion Criteria Applied", ["DICOM format", "Sufficient image quality", "No severe artifacts"]),
        ("Class Labeling", ["Class 0 = cat · Class 1 = dog", "(from folder structure)", ""]),
    ]
    for i, (title, lines) in enumerate(cols):
        x = MX + 1.5 + i * (cw + 1)
        ty = y0 + height - TITLE_BAR - 1.0
        ax.text(x, ty, title, ha="left", va="top", fontsize=9, fontweight="bold", color=PALETTE["text"])
        ty -= LINE
        for ln in lines:
            if ln:
                ax.text(x, ty, ln, ha="left", va="top", fontsize=8.4, color=PALETTE["muted"])
            ty -= LINE
    L.y = y0 - PAD
    L.arrow()

    bg, ac = PALETTE["preproc"]
    pre_lines = [
        ("Intensity Normalization", "Min-max → 8-bit grayscale → duplicated to 3-channel RGB"),
        ("Resize & Center Crop", "Shorter side → 256 px · center crop to 224 × 224 px"),
        ("ImageNet Normalization", "Mean [0.485, 0.456, 0.406] · Std [0.229, 0.224, 0.225]"),
        ("Data Augmentation", "Rotation ±10° · horizontal flip · brightness · Gaussian noise"),
    ]
    height = TITLE_BAR + PAD + len(pre_lines) * (LINE + 0.35) + PAD
    y0 = L.y - height
    ax.add_patch(FancyBboxPatch((MX, y0), MW, height, boxstyle="round,pad=0.015", facecolor=bg, edgecolor=ac, linewidth=2, clip_on=False))
    ax.add_patch(Rectangle((MX, y0 + height - TITLE_BAR), MW, TITLE_BAR, facecolor=ac, edgecolor="none"))
    ax.text(MX + 1.2, y0 + height - TITLE_BAR / 2, "2 · PREPROCESSING", ha="left", va="center", fontsize=10, fontweight="bold", color="white")
    ty = y0 + height - TITLE_BAR - 1.0
    for title, body in pre_lines:
        ax.text(MX + 1.5, ty, title, ha="left", va="top", fontsize=9, fontweight="bold", color=PALETTE["text"])
        ty -= LINE
        ax.text(MX + 2.5, ty, body, ha="left", va="top", fontsize=8.4, color=PALETTE["muted"])
        ty -= LINE + 0.35
    L.y = y0 - PAD
    L.arrow()

    L.split_row(
        "70 / 15 / 15 patient-level stratified split",
        [
            ("Training Set", "434 images", "130 patients", "70%", PALETTE["split"][0]),
            ("Validation Set", "69 images", "28 patients", "15%", PALETTE["split"][1]),
            ("Test Set", "89 images", "29 patients", "15%", PALETTE["split"][2]),
        ],
    )
    L.arrow()

    L.model_train_row()
    L.arrow()

    L.two_panels_bottom(
        "5 · RESULTS (held-out test)",
        [
            ("82.0%", "hero"),
            ("test set accuracy", "muted"),
            ("Macro F1 0.816 · Balanced 83.2% · AUC 0.826", "normal"),
            ("Dog F1 0.84 (recall 0.78)", "normal"),
            ("Cat F1 0.79 (recall 0.88)", "normal"),
            ("Peak validation accuracy: 81.2%", "normal"),
        ],
        "6 · GRAD-CAM",
        [
            ("Layer4 activations", "bold"),
            ("Class-discriminative heatmaps", "normal"),
            ("on held-out test radiographs", "normal"),
            ("Overlay on thoracic views", "normal"),
        ],
    )

    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.2)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
