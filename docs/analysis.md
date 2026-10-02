# Key findings

What the data says about Indian domestic flight prices (March-June 2019, 5 routes, 10,462 flights after cleaning), and where the price model can and cannot be trusted.

Every number below is printed by `python -m src.analysis` and stored in [models/analysis.json](../models/analysis.json).

**Method.** A raw gap between two medians mixes different routes and airlines. To separate the effect of one attribute, flights are grouped so that the other attributes are identical, the two sides are compared inside each group (only groups with at least 20 flights on each side), and the within-group differences are averaged, weighted by the number of flights. This is a descriptive like-for-like comparison, not a causal estimate.

| Comparison | Raw gap of medians | Like-for-like gap | Held constant | Groups |
| --- | --- | --- | --- | --- |
| Weekend vs weekday | +7.1% | +6.6% | route, airline, stops, month | 40 |
| 1 stop vs non-stop | +119.9% | +55.6% | route, airline | 10 |
| "In-flight meal not included" vs standard fare | +30.0% | -26.5% | route, airline, stops | 7 |
| "No check-in baggage included" vs standard fare | -51.3% | -1.4% | route, airline, stops | 4 |

---

## 1. A fare without a meal looks 30% dearer, and is in fact 26% cheaper

Across the whole dataset, tickets marked "In-flight meal not included" have a median price 2,369 INR (+30.0%) above tickets with no remark. That is a mix effect: 1,830 of the 1,926 tickets with this remark are Jet Airways, the airline with the highest median price in the data.

Comparing the same airline, route and number of stops reverses the sign. The no-meal fare is cheaper in all 7 comparable groups, by 3,453 INR (-26.5%) on average, ranging from -16.1% (Multiple carriers, Delhi → Cochin, 1 stop) to -45.4% (Jet Airways, Banglore → Delhi, 1 stop).

The opposite happens with "No check-in baggage included". The raw gap is -51.3%, but all 318 tickets with this remark are SpiceJet, the airline with the lowest median price. Like for like, the fare without baggage is only 45 INR (-1.4%) cheaper, in all 4 comparable groups (all SpiceJet non-stop).

**So what.** `Additional_Info` identifies the fare product, and its effect can only be read within an airline. This is why adding it lifts the model's cross-validated R² from 0.81 to 0.93 (see the ablation in the README): it separates fares that share the same airline, route and schedule.

## 2. One stop costs more than non-stop, but half of the raw gap is the route mix

The raw gap between 1-stop and non-stop flights is +5,595 INR (+119.9%). Much of it comes from where each kind of flight operates: Chennai → Kolkata has only non-stop flights, while Delhi → Cochin has 3,185 one-stop flights and 213 non-stop.

Within the same route and airline the gap is +3,411 INR (+55.6%). It is positive in all 10 comparable groups, from +265 INR (IndiGo, Kolkata → Banglore) to +6,326 INR (Jet Airways, Banglore → Delhi).

![Extra cost of one stop, same airline and route](images/analysis_stops.png)

**So what.** In this market a connecting flight is a dearer product than a direct one, not a discount option, and the size of the premium depends heavily on the airline. The comparison covers 4,598 of the 9,100 one-stop and non-stop flights; the rest sit in groups too small to compare.

## 3. The "weekend premium" is a March effect

The app's headline number, weekend flights +7.1% dearer, barely changes when route, airline, stops and month are held constant (+6.6%). But the average hides the pattern: the weekend is dearer in only 19 of 40 groups.

| Month | Like-for-like weekend gap | Groups | Groups where the weekend is dearer |
| --- | --- | --- | --- |
| March | +20.6% | 11 | 8 |
| April | +2.8% | 5 | 2 |
| May | +1.5% | 10 | 4 |
| June | +2.6% | 14 | 5 |

**So what.** Outside March there is no reliable weekend premium in this data. The data cannot say why March differs (there is no booking date and only one year), so "weekends cost 7% more" should not be presented as a general rule.

## 4. The model is accurate on typical fares; a small tail carries the error

On the hold-out test set (2,093 flights) the mean absolute error is 681 INR, but the median absolute error is 294 INR (3.9% of the price). 65.5% of predictions are within 500 INR. The worst 5% of rows account for 38.7% of the total error.

| Actual price (INR) | Test rows | MAE (INR) | Mean error (INR) |
| --- | --- | --- | --- |
| < 5,000 | 491 | 293 | +153 |
| 5,000-10,000 | 783 | 649 | +264 |
| 10,000-15,000 | 663 | 601 | -163 |
| 15,000-23,090 | 135 | 1,250 | -898 |
| > 23,090 (outliers) | 21 | 9,864 | -9,864 |

![Test error by route and by actual price](images/analysis_errors.png)

- **Expensive tickets are underpredicted.** Fares above the training outlier fence are missed by 9,864 INR on average, always on the low side: the model never saw prices that high. The 15,000-23,090 band is also underpredicted, by 898 INR on average. Part of this is expected when error is grouped by the actual price, since any model pulls extreme values toward the average.
- **By route**, Banglore → Delhi is the weakest (MAE 904 INR, underpredicted by 313 INR on average); Chennai → Kolkata, which has only non-stop flights, is the easiest (MAE 287 INR).
- **By airline**, "Multiple carriers" is the hardest (MAE 1,344 INR over 259 test rows) and SpiceJet the easiest (MAE 209 INR over 164 rows). "Multiple carriers" is a label for itineraries that combine airlines, so it hides the information that drives the price.
- **Some error cannot be removed with these features.** 450 flights share every model feature with another flight and still have a different price. On those rows, even predicting the group's own median misses by 1,100 INR on average. The missing driver is most likely the booking date.

**So what.** The app's "± 681 INR" understates the accuracy for a typical ticket and badly overstates it for expensive ones. A price-dependent range would be more honest than a single number.

## 5. Predicting the next month is harder than the headline score suggests

The headline evaluation splits rows at random, so the test flights come from the same weeks as the training flights. Training on March-May (7,058 flights) and testing on June (3,311 flights) asks the harder question.

| Evaluation | XGBoost R² | XGBoost MAE (INR) | Baseline R² | Baseline MAE (INR) |
| --- | --- | --- | --- | --- |
| Random 80/20 split, all test rows | 0.8722 | 681 | 0.5468 | 1,851 |
| Random 80/20 split, prices up to 23,090 INR | 0.9294 | 588 | - | - |
| Train March-May, test June | 0.8457 | 1,091 | 0.6954 | 1,544 |

The baseline predicts the training-set median price of the same airline, route and number of stops.

![Test MAE: random split vs June hold-out](images/analysis_temporal.png)

June contains no price above the outlier fence, so the fair comparison for the June result is the in-range row: MAE rises from 588 to 1,091 INR (+85%) and R² falls from 0.929 to 0.846. The model still beats the baseline in June (1,091 vs 1,544 INR), but its advantage shrinks from 1,170 INR to 453 INR.

**So what.** The model is good at filling in prices inside a period it has seen and noticeably weaker at forecasting a new month. Monthly price levels move a lot (median 9,769 INR in March, 5,073 in April, 8,662 in May, 8,510 in June), and four months of one year are not enough to learn a seasonal pattern.

---

## Data quality notes

- 221 of 10,683 raw rows are removed by cleaning: 220 exact duplicates and 1 row with missing values.
- Two pairs of labels mean the same thing and are merged: "New Delhi" / "Delhi" and "No Info" / "No info".
- In 3 rows the arrival time does not equal departure time plus duration, and 1 row has a duration under 30 minutes. They are kept: the rows are too few to matter and there is no way to tell which field is wrong.
- 4 airlines and 6 `Additional_Info` values have fewer than 30 flights each, so nothing can be concluded about them.
- April has only 1,078 flights, against 2,678 to 3,395 in the other months.

## Limits of this analysis

- The comparisons are like-for-like on the listed attributes only. Departure time, duration and booking date are not held constant.
- Groups with fewer than 20 flights on either side are left out, so each comparison covers only part of the data (the counts are in `models/analysis.json`).
- One year, four months, five routes: none of these findings should be assumed to hold for other periods or routes.
