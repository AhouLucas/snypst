"""InkML parsing for the MathWriting corpus.

Each ``.inkml`` carries ``<annotation>`` metadata and ``<trace>`` elements. The
ground-truth label is ``normalizedLabel`` (compact LaTeX), falling back to
``label`` for the single-glyph ``symbols/`` inks which lack a normalized form.

Traces hold ``X Y T`` point tuples (comma-separated); only X/Y are kept.
Coordinates are x-rightward, y-downward (already image orientation).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_NS = "{http://www.w3.org/2003/InkML}"
Point = tuple[float, float]
Stroke = list[Point]


@dataclass(frozen=True)
class Ink:
    sample_id: str
    label: str
    strokes: list[Stroke]

    @property
    def n_points(self) -> int:
        return sum(len(s) for s in self.strokes)


def _parse_trace(text: str) -> Stroke:
    points: Stroke = []
    for chunk in text.split(","):
        parts = chunk.split()
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    return points


def parse_inkml(path: str | Path) -> Ink:
    """Parse one InkML file into an :class:`Ink` (label + strokes)."""
    root = ET.parse(path).getroot()
    annotations: dict[str, str] = {}
    for ann in root.findall(f"{_NS}annotation"):
        key = ann.get("type")
        if key is not None and ann.text is not None:
            annotations[key] = ann.text

    label = annotations.get("normalizedLabel") or annotations.get("label") or ""
    sample_id = annotations.get("sampleId") or Path(path).stem

    strokes = [
        pts
        for trace in root.findall(f"{_NS}trace")
        if trace.text and (pts := _parse_trace(trace.text))
    ]
    return Ink(sample_id=sample_id, label=label.strip(), strokes=strokes)
