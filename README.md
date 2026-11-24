# traffic-harvest
## Data split (locked Nov 24, 2026)
- Train: weekdays Nov 12, 13, 16-20
- Test: Nov 23, 24, Nov 30, Dec 1-3
- Excluded: weekends, Nov 25-27 (Thanksgiving)
- Data freeze: end of Dec 3
- Window: 12:00-17:50 ET, 10-minute slots
- Metric: mean absolute error (minutes), per pair and slot
- Baseline: same pair and slot, average over training days
