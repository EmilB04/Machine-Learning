"""Configurable multilayer perceptron for the Project 2 experiments (E2-E4).

PurchaseMLP in model.py is the fixed baseline (73-64-32-1 with ReLU).
ConfigurableMLP builds exactly the same network with its default arguments,
but the number of hidden layers, their widths, the activation function and
dropout can be changed for the experiments.
"""

from torch import nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class ConfigurableMLP(nn.Module):
    """Feedforward network for binary classification that outputs one logit.

    Args:
        input_dim: Number of input features after preprocessing (73).
        hidden_sizes: Width of each hidden layer, e.g. (64, 32) for two layers.
        activation: "relu", "sigmoid" or "tanh", used after every hidden layer.
        dropout: Dropout probability after every hidden layer (0 = no dropout).
    """

    def __init__(self, input_dim, hidden_sizes=(64, 32), activation="relu", dropout=0.0):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Use one of {list(ACTIVATIONS)}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1).")

        # The Linear layers are created in the same order as in PurchaseMLP.
        # With the same seed, the default arguments therefore give exactly
        # the same initial weights as Andreas's baseline.
        layers = []
        previous_width = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(previous_width, width))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_width = width

        # One raw logit; the sigmoid is applied inside BCEWithLogitsLoss.
        layers.append(nn.Linear(previous_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def linear_layers(self):
        """Return the Linear layers in order, used for gradient-norm logging."""
        return [module for module in self.network if isinstance(module, nn.Linear)]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
