# Step 3 analysis: how to run

From the repo root (the folder that contains data/ and analysis/):

    pip3 install numpy pandas scipy matplotlib
    python3 analysis/fit_model.py          # main analysis, prints sections 1-7
    python3 analysis/make_figures.py       # saves results/ table + 2 charts

Files
- congestion_model.py : the model, fitting, baselines, scoring, bootstrap
- fit_model.py        : main analysis (verification, fit, held-out score,
                        shared-beta test, t0 sensitivity, bootstrap, weekend check)
- make_figures.py     : results/fit_results.csv, results/model_vs_baseline.png,
                        results/mae_by_pair.png

Reported model: t = t0 * (1 + A * r^beta), one A per pair, one shared beta,
t0 = 5th percentile of training readings, fit by least squares on log(t/t0).
Split (locked in README): train Nov 12, 13, 16-20; test Nov 23, 24, 30, Dec 1-3.

Final run on Dec 4 (after data freeze):
    python3 analysis/fit_model.py --boot 200 > results/final_output.txt
    python3 analysis/make_figures.py
