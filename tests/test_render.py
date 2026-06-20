from PIL import Image

from dataset.render import RenderConfig, render_strokes, render_to_png


def test_render_dimensions_and_mode():
    strokes = [[(0.0, 0.0), (100.0, 50.0)], [(0.0, 50.0), (100.0, 0.0)]]
    cfg = RenderConfig(target_height=64)
    img = render_strokes(strokes, cfg)
    assert img.mode == "L"
    assert img.height == 64
    assert img.width >= 2 * cfg.padding + 1


def test_render_draws_ink():
    # A diagonal stroke must leave dark pixels on the white canvas.
    img = render_strokes([[(0.0, 0.0), (100.0, 100.0)]])
    assert img.getextrema()[0] == 0  # at least one black pixel


def test_empty_strokes_safe():
    img = render_strokes([])
    assert img.size[0] > 0 and img.size[1] > 0


def test_render_to_png(tmp_path):
    out = tmp_path / "sub" / "x.png"
    render_to_png([[(0.0, 0.0), (10.0, 10.0)]], out)
    assert out.exists()
    assert Image.open(out).format == "PNG"
