# Project 2: Experiment log (observations for the report)

Running notes on observations from the MLP experiments that should be documented in the report (Sections 4.3, 6 and 7.2). Each entry states what was seen, under which run, and what still has to be checked.

## 2026-10-22: Collapsed run in the quick smoke test (E2)

- **Run:** `python experiments.py --stage all --quick` (2 seeds, max 5 epochs, patience 3). Not a real result; smoke test only.
- **Observation:** L3-W16 (3 hidden layers: 16-8-4, ReLU, Adam lr 0.001), **seed 1**, reached validation AP **0.157**, with best epoch 3/5. Seed 0 of the same configuration reached 0.601.
- **Interpretation:** 0.157 equals the purchase rate in the validation set (381/2,441 = 0.156). An AP equal to the positive-class prevalence means the network ranked sessions no better than chance, i.e. it gave nearly the same probability to every session and learned nothing useful.
- **Likely cause (to verify):** a narrow, deep ReLU network with unlucky initial weights, where many units output 0 for all inputs ("dead ReLUs") and therefore receive no gradient. The other plausible explanation is a flat region or poor local optimum.
- **Why it matters for the report:** the assignment asks "Do you see signs of local optima? The error valley problem? Vanishing gradients?" A seed-dependent collapse of the same configuration is concrete evidence of sensitivity to initialization and possible poor optima.
- **Checked in the real E2 run:** the collapse did not recur (see below).

## 2026-10-22: E2 architecture results (real run)

- **Run:** `python experiments.py --stage e2`. 12 architectures × 5 seeds (0–4), Adam lr 0.001, batch 64, max 200 epochs, early stopping on validation AP with patience 20. Total runtime about 12 minutes. Files: `e2_runs.csv`, `e2_summary.csv`, `e2_histories.json`, `e2_selected.json`.
- **Summary (mean validation AP ± SD over 5 seeds):**

| Config | Hidden | Mean AP | SD | Mean best epoch | Params |
|---|---|---|---|---|---|
| L2-W32 | 32-16 | 0.6981 | 0.0040 | 17.6 | 2,913 |
| L3-W32 | 32-16-8 | 0.6978 | 0.0050 | 22.6 | 3,041 |
| L2-W64 | 64-32 (baseline) | 0.6959 | 0.0043 | 13.0 | 6,849 |
| L1-W128 | 128 | 0.6955 | 0.0017 | 16.4 | 9,601 |
| L1-W64 | 64 | 0.6954 | 0.0043 | 19.4 | 4,801 |
| L1-W32 | 32 | 0.6943 | 0.0049 | 31.2 | 2,401 |
| L3-W64 | 64-32-16 | 0.6938 | 0.0054 | 13.4 | 7,361 |
| L1-W16 | 16 | 0.6929 | 0.0024 | 42.2 | 1,201 |
| L2-W16 | 16-8 | 0.6907 | 0.0140 | 40.2 | 1,329 |
| L2-W128 | 128-64 | 0.6885 | 0.0047 | 7.6 | 17,793 |
| L3-W128 | 128-64-32 | 0.6870 | 0.0032 | 9.0 | 19,841 |
| L3-W16 | 16-8-4 | 0.6828 | 0.0210 | 47.0 | 1,361 |

- **Selected (pre-registered rule):** L1-W32 (one hidden layer of 32 units, 2,401 parameters), mean AP 0.6943 ± 0.0049. Rule: fewest parameters among configurations whose mean AP is at least the best mean minus one SD of the best (0.6981 − 0.0040 = 0.6942); six configurations qualified. L1-W32 qualified by a margin of 0.0001, which should be stated honestly.
- **Finding 1: architecture matters little.** All 12 means lie between 0.683 and 0.698 (range 0.015); the top six are within one SD of each other. Depth and width are not the limiting factor for this dataset at this budget.
- **Finding 2: H2 is supported.** Three hidden layers never beat the best two-layer network (L3-W32 0.6978 vs L2-W32 0.6981), and the largest networks (L2-W128, L3-W128) are among the weakest.
- **Finding 3: capacity versus epochs (overfitting).** Wider networks reach their best validation AP much earlier and at a lower level: W128 networks peak at epochs 7–10, W16 networks at epochs 40–47. Larger networks fit the training data faster and start overfitting sooner; this links directly to "How is overfitting related to the various choices?" and to the number of epochs.
- **Finding 4: seed sensitivity of narrow deep networks.** L3-W16 has the largest spread (SD 0.021; seeds range 0.649–0.701) and some seeds need up to 92 epochs. L2-W16 also varies (SD 0.014; 0.670–0.705). The complete collapse to AP ≈ 0.157 from the quick test did **not** recur with longer training, so it was most likely an early plateau that more epochs escaped, not a permanent dead network. Still worth inspecting the gradient norms for L3-W16 seed 1 (AP 0.649) in `e2_histories.json`.
- **Baseline check:** the baseline architecture (L2-W64) reaches 0.6959 ± 0.0043 with patience 20, consistent with 0.6975 for seed 42 with patience 5.
