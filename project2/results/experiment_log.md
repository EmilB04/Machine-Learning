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

- **Run:** `python experiments.py --stage e2`. 12 architectures × 5 seeds (0–4), Adam lr 0.001, batch 64, max 200 epochs, early stopping on validation AP with patience 20. Total runtime about 12 minutes. Files: `results/e2_runs.csv`, `e2_summary.csv`, `e2_histories.json`, `e2_selected.json`.
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

## 2026-10-22: E3 in progress: epoch cap reached at low learning rate

- **Run:** `python experiments.py --stage e3` on L1-W32 (selected in E2), max 200 epochs, patience 20.
- **Observation:** the first run, Adam lr 0.0001 seed 0, reached AP 0.6916 with **best epoch 200/200** (54.6 s). Validation AP was still improving when training hit the 200-epoch cap, so the run stopped because of the cap, not because of early stopping.
- **Implication:** for the lowest learning rates (Adam 1e-4 and probably SGD with low lr and low momentum), the reported AP underestimates what the configuration can reach; these runs are not converged. Compare `best_epoch` with `epochs_run` in `e3_runs.csv`: rows where best_epoch equals 200 are cap-limited.
- **Resolved in the E3 results below:** the cap-limited settings all have low effective learning rate and are not re-run.

## 2026-10-22: E3 optimizer results (real run)

- **Run:** `python experiments.py --stage e3` on L1-W32 (selected in E2), 5 seeds, batch 64, max 200 epochs, patience 20. Grid: Adam lr {1e-4, 3e-4, 1e-3, 3e-3}; SGD momentum {0, 0.5, 0.9} × lr {0.003, 0.01, 0.03, 0.1}; plus a cosine-schedule variant of the best setting (no early stopping, epoch budget = round(mean epochs run of the best)). Files: `results/e3_runs.csv`, `e3_summary.csv`, `e3_histories.json`, `e3_selected.json`.
- **Summary (mean validation AP over 5 seeds), with effective learning rate lr/(1 − momentum):**

| Config | Effective lr | Mean AP | Note |
|---|---|---|---|
| SGD-m0-lr0.1 | 0.1 | **0.7084** ± 0.0050 | selected; mean best epoch 43.6, mean epochs run 63.6 |
| SGD-m0.9-lr0.01 | 0.1 | 0.7070 ± 0.0047 | |
| SGD-m0.5-lr0.03 | 0.06 | 0.7067 ± 0.0059 | |
| SGD-m0.5-lr0.1 | 0.2 | 0.7065 | |
| SGD-m0.9-lr0.03 | 0.3 | 0.7058 | |
| SGD-m0-lr0.03 | 0.03 | 0.7053 | mean best epoch ≈ 127 |
| SGD-m0.9-lr0.003 | 0.03 | 0.7052 | |
| SGD-m0.5-lr0.01 | 0.02 | 0.7048 ± 0.0050 | partly cap-limited (3 seeds best at epoch 193–199) |
| SGD-m0-lr0.1 + cosine | 0.1 → 0 | 0.7038 ± 0.0037 | fixed budget, no early stopping |
| SGD-m0.9-lr0.1 | 1.0 | 0.7033 | too large a step |
| Adam-lr0.003 | – | 0.6964 | best Adam |
| Adam-lr0.001 | – | 0.6943 | = E2 result for L1-W32 |
| Adam-lr0.0003 | – | 0.6933 | |
| Adam-lr0.0001 | – | 0.6886 | cap-limited |
| SGD-m0-lr0.01 | 0.01 | 0.6877 | cap-limited |
| SGD-m0.5-lr0.003 | 0.006 | 0.6628 | cap-limited |
| SGD-m0-lr0.003 | 0.003 | 0.6378 | cap-limited |

- **Selected (pre-registered rule, highest mean):** SGD, momentum 0, lr 0.1. The top seven SGD settings (0.7052–0.7084) lie within about one SD of each other, so the ranking among them is not meaningful; the report should say "well-tuned SGD" rather than claim that momentum 0 beats momentum 0.9.
- **Finding 1: SGD beats Adam on this problem.** Best SGD 0.7084 vs best Adam 0.6964, a gap of 0.012 AP (about 2.4 SD of the winner), and every converged SGD setting beats every Adam setting. Consistent with Wilson et al. (2017), already cited in Related Work: adaptive methods can find solutions that generalize worse than plain SGD. Within Adam, a larger lr was better (3e-3 > 1e-3 > 3e-4 > 1e-4).
- **Finding 2: momentum and lr act through the effective learning rate lr/(1 − momentum).** Settings with the same effective lr give the same result: SGD-m0-lr0.03 and SGD-m0.9-lr0.003 (both 0.03) reach 0.7053 and 0.7052; SGD-m0-lr0.1 and SGD-m0.9-lr0.01 (both 0.1) reach 0.7084 and 0.7070. Effective lr 0.03–0.3 is a broad plateau (0.705–0.708); 1.0 is too large (0.7033); ≤ 0.02 is too slow for 200 epochs. This answers the assignment question on momentum: momentum mainly rescales the step size here rather than adding a separate benefit.
- **Finding 3: the cosine schedule did not help** (0.7038 vs 0.7084). It runs a fixed number of epochs without early stopping and only reaches small steps near the end; with early stopping already selecting the best epoch, annealing adds nothing.
- **Finding 4: the epoch cap.** Adam 1e-4, SGD-m0-lr0.003, SGD-m0-lr0.01, SGD-m0.5-lr0.003 and partly SGD-m0.5-lr0.01 were limited by the 200-epoch cap; their AP is a lower bound. **No re-run needed:** all of them have effective lr ≤ 0.02, and the converged settings with effective lr 0.03 already plateau at ≈ 0.705, below the winner. More epochs would make them slower routes to the same plateau, not a better optimum.
- **Limitation (sequential tuning):** the architecture (E2) was selected under Adam lr 1e-3, and the optimizer was changed to SGD afterwards. The E2 ranking might differ under SGD. Given how flat the E2 ranking was (range 0.015), this is unlikely to change conclusions, but it must be stated in Section 7.
- **Next:** E4 (dropout × weight decay) on L1-W32 + SGD-m0-lr0.1. With SGD, weight decay is applied as the plain L2 term in torch.optim.SGD (the Adam→AdamW switch does not apply).

## 2026-10-22: E4 regularization results (real run)

- **Run:** `python experiments.py --stage e4` on L1-W32 + SGD (momentum 0, lr 0.1), 5 seeds, max 200 epochs, patience 20. Grid: dropout {0, 0.2, 0.5} × weight decay {0, 1e-4, 1e-3}. For SGD, weight decay is the plain L2 term (no AdamW switch). Total runtime about 15 minutes. Files: `results/e4_runs.csv`, `e4_summary.csv`, `e4_histories.json`, `e4_selected.json`.
- **Summary (mean validation AP ± SD over 5 seeds):**

| Config | Mean AP | SD | Mean best epoch | Mean epochs run |
|---|---|---|---|---|
| drop0-wd0.0001 | **0.7089** | 0.0046 | 51.4 | 71.4 |
| drop0-wd0 | 0.7084 | 0.0050 | 43.6 | 63.6 |
| drop0-wd0.001 | 0.7065 | 0.0046 | 71.4 | 91.4 |
| drop0.2-wd0 | 0.7037 | 0.0023 | 83.8 | 103.8 |
| drop0.2-wd0.0001 | 0.7026 | 0.0026 | 70.4 | 90.4 |
| drop0.2-wd0.001 | 0.6991 | 0.0042 | 94.0 | 114.0 |
| drop0.5-wd0 | 0.6924 | 0.0048 | 91.6 | 111.6 |
| drop0.5-wd0.0001 | 0.6920 | 0.0026 | 78.0 | 98.0 |
| drop0.5-wd0.001 | 0.6873 | 0.0045 | 111.8 | 131.8 |

- **Selected (pre-registered rule, highest mean):** dropout 0, weight decay 1e-4, mean AP 0.7089 ± 0.0046. **The gain over no regularization is not real:** +0.0005 (about 0.1 SD); per seed the differences are −0.0014, −0.0002, −0.0001, +0.0028, +0.0015 (2 of 5 positive). The report must say that regularization did not improve validation AP; the selection follows the rule but is equivalent to the unregularized model.
- **Finding 1: dropout hurts monotonically.** Mean AP drops from 0.708 (p = 0) to 0.703 (p = 0.2) and 0.692 (p = 0.5), and the best epoch moves later (44 → 84 → 92). Interpretation: with one hidden layer of 32 units (2,401 parameters) the network has little spare capacity; dropout removes part of it on every step and adds gradient noise, so the network underfits rather than generalizes better. Dropout 0.2 does lower the seed spread (SD 0.0023 vs 0.0050).
- **Finding 2: weight decay has no effect at 1e-4 and a slight negative effect at 1e-3** (0.7065), which also slows convergence (best epoch 71 vs 44).
- **Finding 3: the main regularizer here is early stopping (plus the small architecture).** Together with E2 (wide networks peak at epoch 7–10 and score lower), this answers "How is overfitting related to the various choices?": overfitting is controlled by capacity and the number of epochs, not by dropout or weight decay.
- **No-early-stopping run (overfitting curve):** the first version was built from the E3 selection (weight decay 0) by mistake and reproduced drop0-wd0 seed by seed. Fixed with a separate stage `e4curve` that loads `e4_selected.json`, and re-run with `python experiments.py --stage e4curve`. The corrected run reproduces drop0-wd0.0001 seed by seed (0.7077, 0.7098, 0.7076, 0.7035, 0.7161; mean 0.7089 ± 0.0046), which confirms the fix.
- **Finding 4: patience 20 was sufficient.** Trained for the full 200 epochs, every seed has the same best epoch as with early stopping (45, 64, 45, 49, 54). No seed found a higher validation AP later than 20 epochs after its peak, so early stopping did not cut off any later improvement. The reported AP of this run is the best epoch's AP (best weights are restored); the overfitting evidence for the report is in `e4_no_early_stopping_histories.json` (training vs validation loss after epoch ~50), to be plotted.
- **Final validation-selected configuration for the refit:** 73-32-1, ReLU, SGD lr 0.1 momentum 0, weight decay 1e-4, no dropout, batch 64; epoch count for the train+val refit from the mean best epoch (≈ 51).
