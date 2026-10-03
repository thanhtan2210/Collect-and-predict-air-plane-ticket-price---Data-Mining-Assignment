import json
import os
import re
import xml.etree.ElementTree as ET

import pytest

from src import make_figures
from src.ablation import ABLATION_PATH
from src.predictor import load_interval_metrics, load_metrics

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def figures(tmp_path_factory):
    """File name -> every piece of text drawn on the figure."""
    paths = make_figures.main(str(tmp_path_factory.mktemp("figures")))
    figures = {}
    for path in paths:
        root = ET.parse(path).getroot()
        assert root.tag == SVG + "svg"
        assert root.find(SVG + "title").text.strip()
        figures[os.path.basename(path)] = ["".join(t.itertext()).strip() for t in root.iter(SVG + "text")]
    return figures


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
