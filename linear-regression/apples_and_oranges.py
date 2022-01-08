import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys

if __name__ == "__main__":

    try:

        # Get epochs from function call argument
        epochs = int(sys.argv[1])

    except Exception as e:

        print(f"Exception: {e}")
        print("Usage: python apples_and_oranges.py <NUMBER OF EPOCHS>")
        sys.exit()

    # Given three values (temperature, rainfall and humidity), can we accurately predict the yield of apples and oranges?

    # yield_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
    # yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2
    # w11, w12, ..., w23 = weights
    # b1, b2 = biases

    # Representation of the inputs and targets in matrices
    # We are using 15 rows of input data in this example

    # Input (temp, rainfall, humidity)
    inputs = np.array(
        [
            [73, 67, 43],
            [91, 88, 64],
            [87, 134, 58],
            [102, 43, 37],
            [69, 96, 70],
            [74, 66, 43],
            [91, 87, 65],
            [88, 134, 59],
            [101, 44, 37],
            [68, 96, 71],
            [73, 66, 44],
            [92, 87, 64],
            [87, 135, 57],
            [103, 43, 36],
            [68, 97, 70],
        ],
        dtype="float32",
    )

    # Targets (apples, oranges)
    targets = np.array(
        [
            [56, 70],
            [81, 101],
            [119, 133],
            [22, 37],
            [103, 119],
            [57, 69],
            [80, 102],
            [118, 132],
            [21, 38],
            [104, 118],
            [57, 69],
            [82, 100],
            [118, 134],
            [20, 38],
            [102, 120],
        ],
        dtype="float32",
    )

    # Convert the numpy arrays to tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    # Print the tensors
    print(f"Inputs tensors:\n {inputs}\n")
    print(f"Targets tensors:\n {targets}\n")

    # Dataset
    train_ds = TensorDataset(inputs, targets)
    train_ds[0:3]

    # Dataloader
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Create the model, automatically initializing the weights and biases
    model = nn.Linear(3, 2)
    print(f"Generated initial weights:\n {(model.weight).detach().numpy()}\n")
    print(f"Generated initial biases:\n {(model.bias).detach().numpy()}\n")

    # Generate predictions
    preds = model(inputs)

    # Define the loss function
    loss_fn = F.mse_loss

    # Define the optimizer
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)

    # Utility function to train the model
    def fit(num_epochs, model, loss_fn, opt, train_dl):

        # Repeat for given number of epochs
        for epoch in range(num_epochs):

            # Train with batches of data
            for xb, yb in train_dl:

                # 1. Generate predictions
                pred = model(xb)

                # 2. Calculate loss
                loss = loss_fn(pred, yb)

                # 3. Compute gradients
                loss.backward()

                # 4. Update parameters using gradients
                opt.step()

                # 5. Reset the gradients to zero
                opt.zero_grad()

            # Print the initial loss
            if epoch == 0:
                init_loss = "{:.4f}".format(loss.item())
                print(f"Epoch [0/{num_epochs}], Loss: {init_loss}")

            # Print the progress
            if (epoch + 1) % (num_epochs / 10) == 0:
                epoch_loss = "{:.4f}".format(loss.item())
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")

    # Train the model for 500 epochs
    print(f"Training the model for {epochs} epochs.\n")
    fit(epochs, model, loss_fn, opt, train_dl)

    # Generate the predictions
    preds = model(inputs)

    # Visually compare the predictions and the targets
    print(f"\nPredictions:\n {preds.detach().numpy()}\n")
    print(f"Targets:\n {targets.detach().numpy()}\n")

    # Predict the yields on another data row
    res = model(torch.tensor([[75, 63, 44.0]]))
    x = "{:.4f}".format(res[0][0].item())
    y = "{:.4f}".format(res[0][1].item())
    print(
        f"For a new other data row with 75 temp, 63 rainfall and 44 humidity, the expected apple yield is {x} and the expected orange yield is {y}"
    )

    # Predict the yields on another data row
    res = model(torch.tensor([[62, 53, 43.0]]))
    x = "{:.4f}".format(res[0][0].item())
    y = "{:.4f}".format(res[0][1].item())
    print(
        f"For a new other data row with 62 temp, 53 rainfall and 43 humidity, the expected apple yield is {x} and the expected orange yield is {y}"
    )
