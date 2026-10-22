"""Print the gradient norm per layer for the diag stage (sigmoid vs ReLU).

Reads results/diag_histories.json and prints, for epochs 1, 10 and 100, the
mean L2 norm of each Linear layer's weight gradient, averaged over the seeds.
Layer 1 is the first hidden layer (closest to the input), layer 7 the output.
Vanishing gradients show up as layer 1 being much smaller than layer 7.
"""

import json
from pathlib import Path

import numpy as np

path = Path(__file__).parent / "results" / "diag_histories.json"
histories = json.loads(path.read_text())

for config in ("deep6-sigmoid", "deep6-relu"):
    runs = [h for key, h in histories.items() if key.startswith(config + "|")]
    norms = np.array([run["gradient_norms"] for run in runs])  # seeds x epochs x layers
    print(f"\n{config} ({len(runs)} seeds): mean gradient norm per layer")
    print("epoch  " + "  ".join(f"layer{i + 1:>2}" for i in range(norms.shape[2])) + "   first/last")
    for epoch in (1, 10, 100):
        row = norms[:, epoch - 1, :].mean(axis=0)
        print(f"{epoch:>5}  " + "  ".join(f"{v:8.2e}" for v in row) + f"   {row[0] / row[-1]:9.2e}")
