"""Rasterise InkML strokes to a normalized PNG.

The handwritten source is online (stroke trajectories), but the model consumes
images, so strokes are drawn onto a white canvas. Coordinates are normalized to
a fixed target height (aspect preserved), translated to the origin, and padded.
Stroke width scales with the normalization factor so line weight is consistent
across inks of different raw sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from dataset.inkml import Stroke


@dataclass(frozen=True)
class RenderConfig:
    target_height: int = 128
    padding: int = 8
    stroke_width: int = 3
    max_width: int = 1024  # clamp to avoid pathological aspect ratios


def _bounds(strokes: list[Stroke]) -> tuple[float, float, float, float]:
    xs = [x for s in strokes for x, _ in s]
    ys = [y for s in strokes for _, y in s]
    return min(xs), min(ys), max(xs), max(ys)


def render_strokes(strokes: list[Stroke], cfg: RenderConfig = RenderConfig()) -> Image.Image:
    """Render strokes to a white-background grayscale (``L``) image."""
    strokes = [s for s in strokes if len(s) >= 1]
    if not strokes:
        return Image.new("L", (cfg.target_height, cfg.target_height), color=255)

    min_x, min_y, max_x, max_y = _bounds(strokes)
    raw_h = max(max_y - min_y, 1e-6)
    raw_w = max(max_x - min_x, 1e-6)
    scale = (cfg.target_height - 2 * cfg.padding) / raw_h
    width = min(int(raw_w * scale) + 2 * cfg.padding, cfg.max_width)
    width = max(width, 2 * cfg.padding + 1)

    img = Image.new("L", (width, cfg.target_height), color=255)
    draw = ImageDraw.Draw(img)

    def project(pt: tuple[float, float]) -> tuple[float, float]:
        x, y = pt
        return (
            (x - min_x) * scale + cfg.padding,
            (y - min_y) * scale + cfg.padding,
        )

    for stroke in strokes:
        pts = [project(p) for p in stroke]
        if len(pts) == 1:
            x, y = pts[0]
            r = cfg.stroke_width / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        else:
            draw.line(pts, fill=0, width=cfg.stroke_width, joint="curve")
    return img


def render_to_png(
    strokes: list[Stroke], out_path: str | Path, cfg: RenderConfig = RenderConfig()
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_strokes(strokes, cfg).save(out, format="PNG")
    return out
