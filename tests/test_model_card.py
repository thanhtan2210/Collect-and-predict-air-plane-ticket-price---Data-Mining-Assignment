import json

import pytest

from src import model_card
from src.predictor import known_categories, known_routes, load_intervals, load_pipeline


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    path = tmp_path_factory.mktemp("reports") / "model_card.json"
    model_card.main(str(path))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_airlines_and_routes_match_the_model(card):
    categories = known_categories()
    in_card = {col["name"]: col["values"] for col in card["inputs"]["categorical"]}
    assert in_card == categories

    routes = card["inputs"]["routes"]
    assert routes == known_routes()
    assert set(routes) == set(categories["Source"])
    assert {dest for dests in routes.values() for dest in dests} == set(categories["Destination"])


def test_hyperparameters_match_get_params(card):
    intervals = load_intervals()
    models = {
        "point": load_pipeline().named_steps["model"],
        "lower": intervals["lower"].named_steps["model"],
        "upper": intervals["upper"].named_steps["model"],
    }
    for key, model in models.items():
        params = model.get_params()
        assert card["hyperparameters"][key]
        for name, value in card["hyperparameters"][key].items():
            assert params[name] == value

    assert card["hyperparameters"]["lower"]["quantile_alpha"] == intervals["lower_quantile"]
    assert card["hyperparameters"]["upper"]["quantile_alpha"] == intervals["upper_quantile"]


def test_committed_model_card_is_up_to_date(card):
    with open(model_card.MODEL_CARD_PATH, encoding="utf-8") as f:
        committed = json.load(f)
    # The commit ids depend on the checkout (CI clones without history).
    for section in ("inputs", "outputs", "hyperparameters", "training_data"):
        assert committed[section] == card[section]
