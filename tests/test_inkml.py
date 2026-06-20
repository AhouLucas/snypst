import textwrap

from dataset.inkml import parse_inkml

_SAMPLE = textwrap.dedent(
    """\
    <ink xmlns="http://www.w3.org/2003/InkML">
      <annotation type="label">\\frac{a}{b}</annotation>
      <annotation type="normalizedLabel">\\frac{a}{b}=c</annotation>
      <annotation type="sampleId">deadbeef00000000</annotation>
      <trace id="0">0 0 0.0,1 2 10.0,2 4 20.0</trace>
      <trace id="1">5.5 6.5 30.0,6 7 40.0</trace>
    </ink>
    """
)

_SYMBOL = textwrap.dedent(
    """\
    <ink xmlns="http://www.w3.org/2003/InkML">
      <annotation type="label">*</annotation>
      <annotation type="sampleId">cafe000000000000</annotation>
      <trace id="0">1 1 0.0,2 2 5.0</trace>
    </ink>
    """
)


def test_prefers_normalized_label(tmp_path):
    p = tmp_path / "a.inkml"
    p.write_text(_SAMPLE)
    ink = parse_inkml(p)
    assert ink.label == r"\frac{a}{b}=c"
    assert ink.sample_id == "deadbeef00000000"
    assert len(ink.strokes) == 2
    assert ink.strokes[0] == [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    assert ink.n_points == 5


def test_falls_back_to_label(tmp_path):
    p = tmp_path / "s.inkml"
    p.write_text(_SYMBOL)
    ink = parse_inkml(p)
    assert ink.label == "*"
    assert len(ink.strokes) == 1
