# Variable dictionary

## RFM 3 (recent activity)
| Variable | Definition | UCI unit | CDNOW unit |
| --- | --- | --- | --- |
| recency90 | Days since the last purchase within the 90-day window | days | days |
| frequency90 | 90-day purchase count | distinct customer-invoice pairs | retained positive records |
| monetary90 | 90-day gross positive amount | dataset currency | dataset currency |

## Extended 10 / Common 8
| Variable | Definition | UCI | CDNOW |
| --- | --- | --- | --- |
| tenure | Days from first observed positive transaction to cutoff | days | days |
| frequency30 | 30-day purchase count | distinct customer-invoice pairs | retained positive records |
| monetary30 | 30-day gross positive amount | currency | currency |
| products90 | Distinct products in 90 days | count | - (unavailable) |
| mean_order_value90 | monetary90 / frequency90 | currency per invoice | currency per record |
| items_per_order90 | Mean distinct products per invoice | count | - (unavailable) |
| mean_gap90 | Mean spacing between distinct purchasing days (single-day default 90) | days | days |

## Process 3 (BG/NBD history summaries)
| Variable | Definition |
| --- | --- |
| x | Distinct purchasing days minus one |
| t_x | Span from first to latest purchasing day (days) |
| T | Age from first purchasing day to cutoff (days) |

## Notes
- Logistic models: positive columns are transformed by log(1+x) then standard-scaled; median imputation and scaling are fitted on historical training rows. Tree estimators use raw numeric inputs.
- CDNOW common 8 omits the two product-breadth variables (products90, items_per_order90); its frequency counts retained positive records (identical rows retained).
- A single purchasing day within 90 days sets mean_gap90 = 90.
