import pandas as pd
import numpy as np
import xgboost as xgb
import os
import datetime
# Xác định đường dẫn gốc của script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR, 'models', 'best_flight_price_model_optimized.json')
DATA_PATH = os.path.join(
    BASE_DIR, 'data', 'data_for_assignment', 'cleaned_flight_data.csv')


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_prediction(input_data):
    # 1. Kiểm tra file model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại: {MODEL_PATH}")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    # 2. Lấy danh sách cột
    df_sample = pd.read_csv(DATA_PATH, nrows=1)
    feature_columns = df_sample.drop('Price', axis=1).columns

    # 3. Tạo vector input
    input_vector = pd.DataFrame(
        np.zeros((1, len(feature_columns))), columns=feature_columns)

    # 4. Điền Numerical Data
    input_vector['Total_Stops'] = input_data['stops']
    input_vector['Date'] = input_data['date'].day
    input_vector['Month'] = input_data['date'].month
    input_vector['year'] = input_data['date'].year
    input_vector['day_of_week'] = input_data['date'].weekday()
    input_vector['is_weekend'] = 1 if input_data['date'].weekday() >= 5 else 0
    input_vector['Dep_hour'] = input_data['dep_time'].hour
    input_vector['Dep_Minute'] = input_data['dep_time'].minute
    input_vector['Arrival_hour'] = input_data['arr_time'].hour
    input_vector['Arrival_Minute'] = input_data['arr_time'].minute
    # 5. Điền One-Hot Encoding
    airline_col = f"Airline_{input_data['airline']}"
    source_col = f"Source_{input_data['source']}"
    dest_col = f"Destination_{input_data['dest']}"

    for col in [airline_col, source_col, dest_col]:
        if col in input_vector.columns:
            input_vector[col] = 1.0
    # 6. Dự đoán
    prediction = model.predict(input_vector)[0]
    return max(0, prediction)  # Tránh giá âm nếu có


def select_option(options, label):
    print(f"\nSelect {label}:")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        try:
            choice = int(input(f"Enter number (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice-1]
        except:
            pass
        print("Invalid choice. Try again.")


def main():
    clear_screen()
    print("="*60)
    print("      INDIAN FLIGHT PRICE PREDICTOR - FINAL ASSIGNMENT      ")
    print("="*60)
    # Danh sách chuẩn từ dataset
    airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
                'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia']
    sources = ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
    destinations = ['New Delhi', 'Banglore',
                    'Cochin', 'Kolkata', 'Hyderabad', 'Delhi']
    try:
        airline = select_option(airlines, "Airline")
        source = select_option(sources, "Source City")
        dest = select_option(destinations, "Destination City")

        print("\n" + "-"*30)
        date_str = input("Date of Journey (DD/MM/YYYY): ")
        journey_date = datetime.datetime.strptime(date_str, "%d/%m/%Y")

        dep_str = input("Departure Time (HH:MM): ")
        dep_time = datetime.datetime.strptime(dep_str, "%H:%M").time()

        arr_str = input("Arrival Time (HH:MM): ")
        arr_time = datetime.datetime.strptime(arr_str, "%H:%M").time()

        stops = int(input("Total Stops (0-4): "))
        print("\n[Computing] Analyzing data patterns...")

        price = get_prediction({
            'airline': airline, 'source': source, 'dest': dest,
            'date': journey_date, 'dep_time': dep_time, 'arr_time': arr_time,
            'stops': stops
        })

        print("\n" + "X"*60)
        print(f" PREDICTED TICKET PRICE: {price:,.2f} INR")
        print(f" Approximately: {price*295:,.0f} VND")  # Quy đổi tham khảo
        print("X"*60)

    except Exception as e:
        print(f"\n[Error]: {e}")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
