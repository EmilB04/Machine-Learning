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

    # Convert raw logits to probabilities between 0 and 1.
    # Model is not trained, so these predictions does not
    # represent actual purchase probabilities.
    probabilities = torch.sigmoid(output)

    print("Raw logits:", output)
    print("Predicted probabilities:", probabilities)

    # Artificial binary labels for testing the loss function.
    # 1 = purchase, 0 = no purchase
    # These labels are purely for testing purposes and do not correspond to real data.
    y = torch.tensor([
        [1.0],
        [0.0],
        [1.0],
        [0.0],
        [1.0],
    ])

    # Binary cross-entropy applied to raw logits.
    # This is more numerically stable than applying a sigmoid followed by BCELoss.
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer updates the model parameters 
    # using gradients computed during backpropagation.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss = criterion(output, y)

    print("Binary cross-entropy loss:", loss.item())

    # Compute gradients of the loss with respect
    # to all trainable weights and biases
    loss.backward()

    # Inspect the gradients of the first linear layer. 
    # These gradients indicate how the loss changes
    # when individual weights are adjusted.
    first_layer = model.network[0]

    print("Weight shape:", first_layer.weight.shape)
    print("Gradient shape:", first_layer.weight.grad.shape)
    print("Mean absolute gradient:", first_layer.weight.grad.abs().mean().item())

    # Store the first layers weights before updating them 
    weights_before = first_layer.weight.detach().clone()

    # Update weights and biases using the gradients. 
    optimizer.step()

    # Measure how much the weights changed
    weight_change = (
        first_layer.weight.detach() - weights_before
    ).abs().mean().item()

    print("Mean absolute weight change:", weight_change)

    # Clear gradients before the next training step. 
    optimizer.zero_grad()

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)