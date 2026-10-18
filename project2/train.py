
"""
Training pipeline for Project 2.

This first version verifies that the preprocessed
dataset can be used to initialize the MLP and
prepare mini-batches for training.

No model training is performed at this stage.
"""
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import prepare_data
from model import PurchaseMLP


def main():
    """
    Load the prepared dataset, initialize the MLP,
    and create a DataLoader for future training.
    """

    # --------------------------------------------------
    # 1. Load and preprocess the dataset
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 2. Initialize the neural network
    # --------------------------------------------------

    # Initialize the MLP using the actual number
    # of preprocessed input features.
    model = PurchaseMLP(input_dim=input_dim)

    print("\nNeural network architecture:")
    print(model)

    # Count parameters that will be updated during training.
    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", total_params)

    # --------------------------------------------------
    # 3. Convert NumPy arrays to PyTorch tensors
    # --------------------------------------------------

    # Create independent float32 tensors for model input.
    X_train = torch.tensor(
        splits.X_train,
        dtype=torch.float32
    )

    # Convert binary labels to float32.
    # Reshape to (N, 1) to match the MLP output shape.
    y_train = torch.tensor(
        splits.y_train,
        dtype=torch.float32
    ).reshape(-1, 1)

    
    # Convert validation data to PyTorch tensors.
    # These observations are not used for training.
    X_validation = torch.tensor(
        splits.X_validation,
        dtype=torch.float32
    )

    y_validation = torch.tensor(
        splits.y_validation,
        dtype=torch.float32
    ).reshape(-1, 1)


    # Verify the tensor dimensions and data types.
    print("\nPyTorch training tensors:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_train dtype:", X_train.dtype)
    print("y_train dtype:", y_train.dtype)

    # --------------------------------------------------
    # 4. Create the training dataset
    # --------------------------------------------------

    # Pair each training observation with its target.
    train_dataset = TensorDataset(X_train, y_train)

    # --------------------------------------------------
    # 5. Create the DataLoader
    # --------------------------------------------------

    # Split training data into mini-batches of 64.
    # shuffle=True randomizes the order each epoch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    # --------------------------------------------------
    # 6. Verify the mini-batches
    # --------------------------------------------------

    # Retrieve the first mini-batch for inspection.
    X_batch, y_batch = next(iter(train_loader))

    print("\nMini-batch information:")
    print("Number of batches:", len(train_loader))
    print("Batch features:", X_batch.shape)
    print("Batch labels:", y_batch.shape)

     # --------------------------------------------------
    # 7. Configure model training
    # --------------------------------------------------

    # Binary cross-entropy loss for binary classification.
    # The model returns logits, so no sigmoid is needed here.
    criterion = torch.nn.BCEWithLogitsLoss()

    # Adam updates weights and biases using gradients.
    # Learning rate is initially set to 0.001.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # --------------------------------------------------
    # 8. Train the model for one epoch
    # --------------------------------------------------

    # Enable training mode.
    model.train()

    total_loss = 0.0

    # One epoch processes all training observations once.
    for X_batch, y_batch in train_loader:

        # Clear gradients from the previous mini-batch.
        optimizer.zero_grad()

        # Forward pass: calculate raw logits.
        logits = model(X_batch)

        # Compare predictions with the actual labels.
        loss = criterion(logits, y_batch)

        # Backpropagation: calculate gradients.
        loss.backward()

        # Update model parameters using Adam.
        optimizer.step()

        # Accumulate the loss weighted by batch size.
        total_loss += loss.item() * X_batch.size(0)

    # Average training loss across all observations.
    average_loss = total_loss / len(train_dataset)

    print("\nTraining results:")
    print("Epochs completed: 1")
    print("Average training loss:", average_loss)

    
    # --------------------------------------------------
    # 9. Evaluate the model on validation data
    # --------------------------------------------------

    model.eval()

    # Disable gradients during evaluation.
    with torch.no_grad():

        # Obtain raw model outputs.
        validation_logits = model(X_validation)

        # Calculate binary cross-entropy loss.
        validation_loss = criterion(
            validation_logits,
            y_validation
        )

        # Convert logits into purchase probabilities.
        validation_probabilities = torch.sigmoid(
            validation_logits
        )

    # Convert tensors to NumPy arrays for scikit-learn.
    y_true = y_validation.cpu().numpy().ravel()
    y_scores = validation_probabilities.cpu().numpy().ravel()

    # Average Precision summarizes the precision-recall
    # relationship across prediction thresholds.
    validation_ap = average_precision_score(
        y_true,
        y_scores
    )

    print("\nValidation results:")
    print("Validation loss:", validation_loss.item())
    print("Validation Average Precision:", validation_ap)

if __name__ == "__main__":
    main()
