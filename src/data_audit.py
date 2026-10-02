"""Data audit: facts about the dataset that limit how it can be interpreted.

Run from the repo root:  python -m src.data_audit
Writes reports/data_audit.json.
"""
import json
import os

import pandas as pd

from src.predictor import load_clean_data
from src.preprocess import BASE_DIR, DATE_FORMAT

REPORT_DIR = os.path.join(BASE_DIR, "reports")
AUDIT_PATH = os.path.join(REPORT_DIR, "data_audit.json")
JET_AIRWAYS_LAST_FLIGHT = "2019-04-17"


def audit(clean):
    df = clean.copy()
    date = pd.to_datetime(df["Date_of_Journey"], format=DATE_FORMAT)
    df["Month"] = date.dt.month
    df["Route"] = df["Source"] + " → " + df["Destination"]
    jet = df["Airline"] == "Jet Airways"

    monthly = df.groupby("Month").size()
    jet_monthly = df[jet].groupby("Month").size().reindex(monthly.index, fill_value=0)
    info_by_airline = (
        df.groupby(["Additional_Info", "Airline"]).size().rename("flights").reset_index()
    )
    route_month = df.groupby(["Route", "Month"]).size().unstack("Month", fill_value=0)

    return {
        "rows": int(len(df)),
        "distinct_journey_dates": int(date.nunique()),
        "jet_airways": {
            "last_flight": JET_AIRWAYS_LAST_FLIGHT,
            "flights": int(jet.sum()),
            "flights_dated_after_last_flight": int((jet & (date > JET_AIRWAYS_LAST_FLIGHT)).sum()),
            "by_month": [
                {
                    "month": int(month),
                    "all_flights": int(monthly[month]),
                    "jet_airways_flights": int(jet_monthly[month]),
                    "jet_airways_share": float(jet_monthly[month] / monthly[month]),
                }
                for month in monthly.index
            ],
        },
        "additional_info_by_airline": info_by_airline.to_dict("records"),
        "flights_by_route_and_month": [
            {"route": route, **{f"month_{m}": int(n) for m, n in row.items()}, "total": int(row.sum())}
            for route, row in route_month.iterrows()
        ],
    }


def main():
    report = audit(load_clean_data())
    jet = report["jet_airways"]

    print(f"{report['rows']:,} flights on {report['distinct_journey_dates']} distinct journey dates")
    print(f"\nJet Airways (last flight {jet['last_flight']}): {jet['flights']:,} flights, "
          f"{jet['flights_dated_after_last_flight']:,} dated after its last flight")
    for row in jet["by_month"]:
        print(f"  month {row['month']}: {row['jet_airways_flights']:>5,} of {row['all_flights']:>5,} "
              f"flights ({row['jet_airways_share']:.1%})")

    print("\nAdditional_Info by airline:")
    table = pd.DataFrame(report["additional_info_by_airline"])
    print(table.pivot(index="Airline", columns="Additional_Info", values="flights")
          .fillna(0).astype(int).T.to_string())

    print("\nFlights by route and month:")
    print(pd.DataFrame(report["flights_by_route_and_month"]).to_string(index=False))

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nSaved {AUDIT_PATH}")


if __name__ == "__main__":
    main()
