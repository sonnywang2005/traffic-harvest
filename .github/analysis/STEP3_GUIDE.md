# Step 3 guide: fit the congestion model and compare it with the baseline

## Which code we use and why
The main analysis is the friend's `congestion_model.py` + `fit_model.py`. `make_figures.py` adds charts and a results table built on that same code, so there is one model and one beta.

The friend's version was chosen because it:
- fits on log(t/t0), so errors count as percentages
- tries several starting points for the fit
- checks itself: it recovers known A and beta from fake data
- gives 90% uncertainty ranges by resampling whole days
- tests the shared-beta choice and the t0 choice
- adds a second baseline (Mapbox's "typical" time)

Its only gap was charts and saved results; `make_figures.py` fills that.

Results on the 1 test day so far (Mon Nov 23):
- beta = 3.28 (90% range 3.14 to 3.52)
- model 0.65 min off vs. baseline 0.61 min off, using 11 numbers instead of 360
- shared beta does as well as a beta per pair (0.648 vs 0.655)
- Mondays run light: both overpredicted Nov 23 (+0.42) and Nov 16 (+0.59), while other weekdays were within ±0.3. That's a Monday effect, not Thanksgiving. Do not change the locked split; report it as a finding.
- The weekend check (section 7) is exploratory, because its traffic scale was tuned on weekend data.

## Steps

### 1. Set up the repo (10 min)
- Upload this whole `analysis/` folder to the GitHub repo: Add file -> Upload files, and type `analysis/` before each file name so the folder is created.
- Do not upload the older single-file `fit_model.py` Claude wrote first.

### 2. Run it locally (5 min)
- Download the repo (green Code button -> Download ZIP) and unzip it.
- In Terminal, from the repo folder (the one containing `data/` and `analysis/`):

      pip3 install numpy pandas scipy matplotlib
      python3 analysis/fit_model.py
      python3 analysis/make_figures.py

- Check: beta about 3.28, model 0.65 vs. baseline 0.61, and a `results/` folder containing `fit_results.csv`, `model_vs_baseline.png` and `mae_by_pair.png`.

### 3. Agree on the reported model (team, 5 min)
Reported: shared-beta model, t0 = 5th percentile of training readings, fit on log(t/t0).
Supporting evidence: sections 1 (verification), 4 (shared beta), 5 (t0 sensitivity), 6 (uncertainty). Section 7 is exploratory.

### 4. Make sure everyone can explain it (30 min)
Be ready to say in plain words:
- why the fit uses logs: errors become percentages, which suits trips of different lengths
- why resample whole days: readings 10 minutes apart on the same afternoon aren't independent
- what the fake-data check proves: the fitting code recovers answers we already know

### 5. Final run on Dec 4 (15 min)
After the Dec 3 data freeze, re-download `data/travel_times.csv`, then:

      python3 analysis/fit_model.py --boot 200 > results/final_output.txt
      python3 analysis/make_figures.py

Commit `results/` to the repo.

### 6. Write-up points (use the Dec 4 numbers)
- fitted beta with its 90% range; each pair's A and peak delay
- model vs. both baselines across all 6 test days, and by day
- the Monday effect
- shared beta holds up; results barely depend on the t0 choice
- the noon overestimate, likely because the 2017 counter is 1.5 miles away

### 7. Hand beta to Step 5
The road add/remove network uses the final beta from the Dec 4 run (about 3.28).
