# Business summary

**Context.** This is a case study of the Indian domestic flight market in 2019: 10,462 fares for March-June 2019 on five routes, from a public dataset. It shows how fare data can support a pricing decision. It is not advice on today's prices.

**User.** A travel-agency employee holding a quote for a customer, who needs to know whether the price is cheap, fair or expensive compared with similar flights, and which alternative would cost less.

**Tool.** The app takes the flight details and the quoted price and returns a price range for comparable flights, with a label: cheap (below the range), fair (inside) or expensive (above). The range is designed to contain 80% of prices; on 2,093 flights the model had never seen it contained 79.8%. Its average width is 2,274 INR, from 1,138 INR for the cheapest quarter of tickets to 3,024 INR for the dearest quarter.

## Three findings

Each comparison is made between flights that are alike in the stated respects. The figure is the median of the differences across groups, and the 95% confidence interval shows how much it depends on which groups could be compared.

1. **A connection costs more than a direct flight.** On the same airline and route, a one-stop flight cost 3,038 INR more than a non-stop one (95% CI 964 to 4,410 INR; 10 airline-route pairs, 4,598 flights). It was dearer in all 10 pairs, but by anything from 265 to 6,326 INR, so the size of the premium depends on the airline.

2. **On Jet Airways, the fare without a meal was a much cheaper fare class.** On the same route and number of stops it cost 3,436 INR less than the standard fare (95% CI 1,830 to 6,132 INR less; 6 route-stops pairs, 3,634 flights). This is the gap between two Jet Airways fare classes, not the price of a meal. A naive comparison across all airlines says the opposite, that no-meal fares are 361 INR dearer on average, only because 1,830 of the 1,926 no-meal fares belong to Jet Airways, an expensive airline.

3. **The cheapest option was a low-cost carrier flying direct, and the gap to the typical fare was large on the long routes.** SpiceJet had the lowest median price on four of the five routes and GoAir on the fifth. On Kolkata → Banglore the cheapest option (SpiceJet non-stop, median 3,873 INR, 248 flights) was 5,472 INR below the route median; on Delhi → Cochin (SpiceJet one stop, median 5,583 INR, 87 flights) it was 4,679 INR below. On Chennai → Kolkata the gap was only 253 INR.

## Recommendations for the agency (as they would have applied in 2019)

- Check every quote against the price range before passing it on, and look for an alternative when it is labelled expensive.
- Offer the direct flight first: it was cheaper in every airline-route pair that could be compared.
- When a customer wants Jet Airways, quote the no-meal fare class alongside the standard one.
- Spend comparison effort where the cheapest option is furthest below the typical fare: Kolkata → Banglore and Delhi → Cochin.

## Limitations

- **2019 data.** Prices, airlines and regulation have changed; Jet Airways, the largest airline in the data, no longer flies.
- **The date column is unreliable.** 2,600 Jet Airways rows are dated after the airline's last flight on 17 April 2019, so the dates may not be actual flight dates.
- **Five routes only**, and each comparison uses only the groups with enough flights (10 and 6 groups), so the confidence intervals are wide.
- **No booking date.** How far ahead a ticket is bought strongly affects its price and is not in the data, which is one reason the price range is wide.

Sources of the figures: [reports/findings.json](../reports/findings.json) (`python -m src.business_analysis`), [models/interval_metrics.json](../models/interval_metrics.json) (`python -m src.train_intervals`), [reports/data_audit.json](../reports/data_audit.json) (`python -m src.data_audit`).
