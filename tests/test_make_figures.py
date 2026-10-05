import json
import os
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont

from src import make_figures
from src.ablation import ABLATION_PATH
from src.predictor import load_interval_metrics, load_metrics
from src.preprocess import BASE_DIR

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def svg_paths(tmp_path_factory):
    return make_figures.main(str(tmp_path_factory.mktemp("figures")))


@pytest.fixture(scope="module")
def figures(svg_paths):
    """File name -> every piece of text drawn on the figure."""
    figures = {}
    for path in svg_paths:
        root = ET.parse(path).getroot()
        assert root.tag == SVG + "svg"
        assert root.find(SVG + "title").text.strip()
        figures[os.path.basename(path)] = ["".join(t.itertext()).strip() for t in root.iter(SVG + "text")]
    return figures


@pytest.fixture(scope="module")
def data():
    return make_figures.load_figure_data()


def read_svg(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("theme", make_figures.THEMES)
@pytest.mark.parametrize("name", make_figures.FIGURES)
def test_drawing_function_returns_a_figure(data, name, theme):
    draw, _ = make_figures.FIGURES[name]
    fig = draw(data, theme)
    assert isinstance(fig, Figure)
    assert fig.get_facecolor() == to_rgba(make_figures.THEMES[theme]["background"])
    plt.close(fig)


def test_svg_files_match_the_committed_ones(svg_paths):
    # Text positions depend on the font the committed files were laid out with.
    try:
        findfont(make_figures.RC["font.sans-serif"][0], fallback_to_default=False)
    except ValueError:
        pytest.skip("the committed SVG files were drawn with Segoe UI, which is not installed here")
    for path in svg_paths:
        committed = os.path.join(make_figures.IMAGE_DIR, os.path.basename(path))
        assert read_svg(path) == read_svg(committed), os.path.basename(path)


def test_captions_are_the_readme_sentences(data):
    with open(os.path.join(BASE_DIR, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    captions = make_figures.captions(data)
    assert set(captions) == set(make_figures.FIGURES)
    for name, caption in captions.items():
        assert caption + " (`python -m src.make_figures`)." in readme, name


def labels(texts, pattern):
    return sorted(text for text in texts if re.fullmatch(pattern, text))


def test_twelve_svg_files(figures):
    expected = {f"{name}_{mode}.svg" for name in make_figures.FIGURES for mode in make_figures.THEMES}
    assert len(expected) == 12
    assert set(figures) == expected


@pytest.mark.parametrize("mode", make_figures.THEMES)
def test_ablation_labels_match_the_json(figures, mode):
    with open(ABLATION_PATH, encoding="utf-8") as f:
        sets = json.load(f)["feature_sets"]
    texts = figures[f"fig_ablation_{mode}.svg"]
    assert labels(texts, r"\d\.\d{4}") == sorted(f"{s['r2_mean']:.4f}" for s in sets)
    assert all(s["feature_set"] in texts for s in sets)


@pytest.mark.parametrize("mode", make_figures.THEMES)
def test_model_labels_match_the_json(figures, mode):
    cv = load_metrics()["cv"]
    texts = figures[f"fig_models_{mode}.svg"]
    assert labels(texts, r"\d\.\d{4} ± \d\.\d{4}") == sorted(
        f"{m['r2_mean']:.4f} ± {m['r2_std']:.4f}" for m in cv.values()
    )
    assert all(name in texts for name in cv)


@pytest.mark.parametrize("mode", make_figures.THEMES)
def test_coverage_labels_match_the_json(figures, mode):
    intervals = load_interval_metrics()
    bands = intervals["test_by_price_quartile"]
    texts = figures[f"fig_interval_coverage_{mode}.svg"]
    assert labels(texts, r"\d+\.\d%") == sorted(f"{b['coverage']:.1%}" for b in bands)
    assert labels(texts, r"mean width [\d,]+ INR") == sorted(
        f"mean width {b['mean_width']:,.0f} INR" for b in bands
    )
    assert f"target {intervals['target_coverage']:.0%}" in texts
