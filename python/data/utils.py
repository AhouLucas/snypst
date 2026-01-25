import pandoc as pdc
import subprocess
import shutil
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_latex_into_valid_expression(latex_formula: str) -> str:
    "Adds dollar signs $ into Latex expression"

    if not latex_formula.startswith("$"):
        return "$" + latex_formula + "$"
    
    return latex_formula

def convert_latex_to_typst(latex_formula: str) -> tuple[str, bool]:
    """
    Convert LaTeX to Typst using Pandoc subprocess.
    Returns (typst_string, failed_flag)
    """

    try:
        result = subprocess.run(
            ["pandoc", "-f", "latex", "-t", "typst"],
            input=latex_formula,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            print(f"Pandoc Error: {result.stderr}")
            return "", True
        
        # Check stderr for warnings (non-fatal but may indicate issues)
        if result.stderr:
            print(f"Pandoc warning: {result.stderr}")
            return "", True
        
        return result.stdout.strip(), False
        
    except subprocess.TimeoutExpired:
        print("Pandoc conversion timed out")
        return "", True
    except FileNotFoundError:
        print("Pandoc not found in PATH")
        return "", True
    


def ensure_empty_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, clear it if it does."""
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def convert_inkml_to_png(inkml_path: str | Path, output_path: str | Path,
                  img_size: tuple[int, int] = (256, 256), 
                  stroke_width: int = 3,
                  padding: int = 10) -> dict:
    """
    Convert InkML file to PNG.
    
    Args:
        inkml_path: Path to the InkML file
        output_path: Path for the output PNG
        img_size: Output image dimensions (width, height)
        stroke_width: Width of rendered strokes (recommended ~1.5 radius = ~3 width)
        padding: Padding around the rendered strokes
    
    Returns:
        Dict with metadata (label, normalizedLabel, sampleId)
    """
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    ns = {'ink': 'http://www.w3.org/2003/InkML'}
    
    # Extract metadata from annotations
    metadata = {}
    for annotation in root.findall('ink:annotation', ns):
        ann_type = annotation.get('type')
        metadata[ann_type] = annotation.text
    
    # Parse all traces (strokes)
    traces = []
    all_points = []
    
    for trace in root.findall('ink:trace', ns):
        # Format: "x y t, x y t, ..." (space-separated x, y, t; comma-separated points)
        points = []
        for point_str in trace.text.strip().split(','):
            parts = point_str.strip().split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                points.append((x, y))
                all_points.append((x, y))
        if points:
            traces.append(points)
    
    if not all_points:
        # Empty ink - create blank image
        img = Image.new('RGB', img_size, 'white')
        img.save(output_path)
        return metadata
    
    # Calculate bounds
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)
    
    # Handle single-point or line cases
    width = max_x - min_x if max_x != min_x else 1
    height = max_y - min_y if max_y != min_y else 1
    
    # Calculate scale to fit in image with padding
    w, h = img_size
    scale_x = (w - 2 * padding) / width
    scale_y = (h - 2 * padding) / height
    scale = min(scale_x, scale_y)
    
    # Center the drawing
    scaled_width = width * scale
    scaled_height = height * scale
    offset_x = (w - scaled_width) / 2
    offset_y = (h - scaled_height) / 2
    
    # Create image (white background)
    img = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw each trace
    for points in traces:
        scaled_points = [
            (offset_x + (x - min_x) * scale, 
             offset_y + (y - min_y) * scale)
            for x, y in points
        ]
        
        if len(scaled_points) == 1:
            # Single point - draw a dot
            x, y = scaled_points[0]
            r = stroke_width / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill='black')
        else:
            # Multiple points - draw connected lines
            draw.line(scaled_points, fill='black', width=stroke_width, joint='curve')
    
    img.save(output_path)
    return metadata


def get_inkml_label(inkml_path: str | Path) -> str:
    """Extract the label (normalizedLabel or label) from an InkML file."""
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    ns = {'ink': 'http://www.w3.org/2003/InkML'}
    
    normalized_label = None
    label = None
    
    for annotation in root.findall('ink:annotation', ns):
        ann_type = annotation.get('type')
        if ann_type == 'normalizedLabel':
            normalized_label = annotation.text
        elif ann_type == 'label':
            label = annotation.text
    
    # Use normalizedLabel if available, otherwise label (for symbols/)
    return normalized_label or label or ""
