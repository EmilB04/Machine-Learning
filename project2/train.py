
"""
Training pipeline for Project 2.

This first version verifies that the preprocessed
dataset can be used to initialize the MLP and
prepare mini-batches for training.

No model training is performed at this stage.
"""
import copy
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

    # Fix the PyTorch random seed for reproducibility.
    torch.manual_seed(42)

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
    # 8. Train and validate after each epoch
    # --------------------------------------------------

    num_epochs = 20

    # Store results for later analysis and plots.
    train_losses = []
    validation_losses = []
    validation_aps = []

    # Track the model with the highest validation AP.
    best_validation_ap = float("-inf")
    best_epoch = 0
    best_model_weights = None

    
    # Stop training after several epochs without improvement.
    patience = 5
    epochs_without_improvement = 0


    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        average_loss = total_loss / len(train_dataset)

        # Evaluate on validation data after the epoch.
        model.eval()

        with torch.no_grad():
            validation_logits = model(X_validation)

            validation_loss = criterion(
                validation_logits,
                y_validation
            )

            validation_probabilities = torch.sigmoid(
                validation_logits
            )

        # Convert to NumPy for scikit-learn AP calculation.
        y_true = y_validation.cpu().numpy().ravel()
        y_scores = (
            validation_probabilities.cpu().numpy().ravel()
        )

        validation_ap = average_precision_score(
            y_true,
            y_scores
        )

      
        # Save the best model according to validation AP.
        if validation_ap > best_validation_ap:
            best_validation_ap = validation_ap
            best_epoch = epoch + 1
            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            # Reset patience when validation AP improves.
            epochs_without_improvement = 0
        else:
            # Count epochs without improvement.
            epochs_without_improvement += 1


        # Save the metrics for this epoch.
        train_losses.append(average_loss)
        validation_losses.append(validation_loss.item())
        validation_aps.append(validation_ap)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train loss: {average_loss:.4f} | "
            f"Val loss: {validation_loss.item():.4f} | "
            f"Val AP: {validation_ap:.4f}"
        )

        
        # Stop when validation AP has not improved
        # for the specified number of epochs.
        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping triggered at "
                f"epoch {epoch + 1}."
            )
            break


    # Restore the best validation checkpoint.
    model.load_state_dict(best_model_weights)

    print("\nBest model:")
    print("Best epoch:", best_epoch)
    print("Best validation AP:", best_validation_ap)

    
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
