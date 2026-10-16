import torch
from torch import nn

class PurchaseMLP(nn.Module):
    """
    Multilayer Perceptron (MLP) for binary classification.

    The network predicts whether an online shopping
    session will result in a purchase.

    Architecture:
        Input -> 64 neurons -> ReLU
              -> 32 neurons -> ReLU
              -> 1 output (logit)

    Args:
        input_dim (int): Number of input features after
                         preprocessing and encoding.
    """
    def __init__(self, input_dim):
        super().__init__()

        # Define a fully connected feedforward network.
        # ReLU introduces non-linearity between layers.

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            # Return a single logit for binary classification.
            # No sigmoid here; use BCEWithLogitsLoss during training for numerical stability.
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    # Temporary test using 17 input features.
    # The actual input dimension will be determined
    # after preprocessing the dataset.
    model = PurchaseMLP(input_dim=17)
    print(model)

    # Count the parameters that will be updated
    # during training (weights and biases).
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    print("Trainable parameters:", total_params)

    # Generate five artificial observations with
    # 17 features each to test the network.
    # These are NOT real observations from the dataset.
    x = torch.randn(5, 17)

    # Run a forward pass without training the model.
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)