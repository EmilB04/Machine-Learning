
"""
Training pipeline for Project 2.

This first version verifies that the preprocessed
dataset can be used to initialize the MLP.
"""

from preprocessing import prepare_data
from model import PurchaseMLP
import torch


def main():
    """
    Load the prepared dataset and initialize the MLP.

    No model training is performed at this stage.
    """

    # Load the stratified train, validation, and test sets.
    # Encoding and scaling are fitted on training data only.
    splits = prepare_data()

    # Determine the actual number of input features
    # after categorical one-hot encoding.
    input_dim = splits.X_train.shape[1]

    # Display the dataset partition sizes.
    print("Training observations:", len(splits.y_train))
    print("Validation observations:", len(splits.y_validation))
    print("Test observations:", len(splits.y_test))
    print("Input features:", input_dim)

    # Initialize the neural network using the
    # actual number of preprocessed input features.
    model = PurchaseMLP(input_dim=input_dim)

    print("\nNeural network architecture:")
    print(model)

    # Count all parameters that require gradients.
    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", total_params)

    
    # Convert NumPy arrays to PyTorch tensors.
    # Features use float32 for neural network calculations.
    X_train = torch.from_numpy(splits.X_train).float()

    # Binary targets must also be floating-point values
    # when using BCEWithLogitsLoss.
    y_train = torch.from_numpy(
        splits.y_train
    ).float().unsqueeze(1)

    # Verify that features and targets have
    # compatible dimensions.
    print("\nPyTorch training tensors:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_train dtype:", X_train.dtype)
    print("y_train dtype:", y_train.dtype)



if __name__ == "__main__":
    main()
