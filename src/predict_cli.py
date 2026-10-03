"""Interactive price prediction in the terminal.

Run from the repo root:  python -m src.predict_cli
"""
from src.predictor import known_categories, known_routes, predict_price
from src.preprocess import STOPS_MAP, parse_duration


def select_option(options, label):
    print(f"\nSelect {label}:")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        try:
            choice = int(input(f"Enter number (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def main():
    print("=" * 60)
    print("      INDIAN FLIGHT PRICE PREDICTOR - FINAL ASSIGNMENT      ")
    print("=" * 60)
    try:
        categories = known_categories()
        routes = known_routes()
    except FileNotFoundError as e:
        print(f"\n[Error]: {e}")
        return

    try:
        airline = select_option(categories["Airline"], "Airline")
        source = select_option(sorted(routes), "Source City")
        dest = select_option(routes[source], "Destination City")

        print("\n" + "-" * 30)
        journey_date = input("Date of Journey (DD/MM/YYYY): ")
        dep_time = input("Departure Time (HH:MM): ")
        duration = parse_duration(input("Flight Duration (e.g. 2h 50m): "))
        stops = int(input(f"Total Stops (0-{max(STOPS_MAP.values())}): "))

        price = predict_price(airline, source, dest, journey_date, dep_time, duration, stops)

        print("\n" + "=" * 60)
        print(f" PREDICTED TICKET PRICE: {price:,.0f} INR")
        print("=" * 60)
    except (ValueError, KeyError) as e:
        print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()
