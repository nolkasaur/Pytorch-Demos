# Program to train image recognition of, in this case, handwritten numbers 0-9, using the MNIST dataset.

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import random
import sys

if __name__ == "__main__":

    try:

        # Get epochs and print tests from function call argument
        epochs = int(sys.argv[1])
        print_tests = int(sys.argv[2])

    except Exception as e:

        print(f"Exception: {e}")
        print(
            "Usage: python MINST_numbers.py <NUMBER OF EPOCHS> <NUMBER_OF_PRINT_TESTS>"
        )
        sys.exit()

    class MnistModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, xb):
            xb = xb.reshape(-1, 784)
            out = self.linear(xb)
            return out

        def training_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            return loss

        def validation_step(self, batch):
            images, labels = batch
            out = self(images)  # Generate predictions
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)  # Calculate accuracy
            return {"val_loss": loss, "val_acc": acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x["val_loss"] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x["val_acc"] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print(
                "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch, result["val_loss"], result["val_acc"]
                )
            )

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        optimizer = opt_func(model.parameters(), lr)
        history = []  # for recording epoch-wise results

        for epoch in range(epochs):

            # Training Phase
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation phase
            result = evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)

        return history

    def evaluate(model, val_loader):
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def predict_image(img, model):
        xb = img.unsqueeze(0)
        yb = model(xb)
        _, preds = torch.max(yb, dim=1)
        return preds[0].item()

    dataset = MNIST(root="data/", train=True, transform=transforms.ToTensor())

    test_dataset = MNIST(root="data/", train=False, transform=transforms.ToTensor())

    train_ds, val_ds = random_split(dataset, [50000, 10000])

    batch_size = 128

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)

    input_size = 28 * 28
    num_classes = 10

    model = MnistModel()

    print(f"Training the model over {epochs} epochs:")
    fit(epochs, 0.001, model, train_loader, val_loader)

    test_loader = DataLoader(test_dataset, batch_size=256)

    acc = int((evaluate(model, test_loader).get("val_acc")) * 100)
    print(f"\nAfter final validation: {acc}% model accuracy.\n")

    print(f"Additional {print_tests} random tests for user visualization:")
    for x in range(print_tests):
        img, label = test_dataset[random.randint(0, len(test_dataset))]
        print("Label:", label, ", Predicted:", predict_image(img, model))

    # For saving progress and posterior uses
    # torch.save(model.state_dict(), "mnist-logistic.pth")
    # model2 = MnistModel()
    # model2.load_state_dict(torch.load('mnist-logistic.pth'))
