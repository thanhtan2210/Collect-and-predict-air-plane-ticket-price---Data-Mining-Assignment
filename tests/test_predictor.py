from src.predictor import known_categories, known_routes, predict_price


def test_routes_match_model_categories():
    routes = known_routes()
    categories = known_categories()
    assert set(routes) == set(categories["Source"])
    destinations = {dest for dests in routes.values() for dest in dests}
    assert destinations == set(categories["Destination"])
    assert "New Delhi" not in destinations


def test_prediction_in_plausible_range():
    price = predict_price("IndiGo", "Banglore", "Delhi", "24/03/2019", "22:20", 170, 0)
    assert 1500 <= price <= 25000


def test_two_stops_cost_more_than_non_stop():
    # Durations are the medians for Jet Airways from Delhi at each stop count.
    non_stop = predict_price("Jet Airways", "Delhi", "Cochin", "15/05/2019", "09:00", 195, 0)
    two_stops = predict_price("Jet Airways", "Delhi", "Cochin", "15/05/2019", "09:00", 1140, 2)
    assert two_stops > non_stop
