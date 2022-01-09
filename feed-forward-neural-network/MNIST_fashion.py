import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import random
import sys

if __name__ == "__main__":

    try:

        # Get epochs from function call argument
        epochs = int(sys.argv[1])
        print_tests = int(sys.argv[2])

    except Exception as e:

        print(f"Exception: {e}")
        print(
            "Usage: python MNIST_fashion.py <NUMBER OF EPOCHS> <NUMBER_OF_PRINT_TESTS>"
        )
        sys.exit()

    dataset = FashionMNIST(root="data/", download=True, transform=ToTensor())
    test_dataset = FashionMNIST(root="data/", train=False, transform=ToTensor())

    val_size = 10000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 128

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size * 2, num_workers=4, pin_memory=True
    )

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class MnistModel(nn.Module):
        """Feedfoward neural network with 1 hidden layer"""

        def __init__(self, in_size, out_size):
            super().__init__()
            # hidden layer
            self.linear1 = nn.Linear(in_size, 16)
            # hidden layer 2
            self.linear2 = nn.Linear(16, 32)
            # output layer
            self.linear3 = nn.Linear(32, out_size)

        def forward(self, xb):
            # Flatten the image tensors
            out = xb.view(xb.size(0), -1)
            # Get intermediate outputs using hidden layer 1
            out = self.linear1(out)
            # Apply activation function
            out = F.relu(out)
            # Get intermediate outputs using hidden layer 2
            out = self.linear2(out)
            # Apply activation function
            out = F.relu(out)
            # Get predictions using output layer
            out = self.linear3(out)
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

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    device = get_default_device()

    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader:
        """Wrap a dataloader to move data to a device"""

        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl:
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    def evaluate(model, val_loader):
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
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

    input_size = 784
    num_classes = 10

    model = MnistModel(input_size, out_size=num_classes)
    to_device(model, device)

    history = [evaluate(model, val_loader)]

    print(f"Training the model over {epochs} epochs with a learning rate of 0.5:")
    history += fit(epochs, 0.5, model, train_loader, val_loader)

    print(f"Training the model over {epochs} epochs with a learning rate of 0.1:")
    history += fit(epochs, 0.1, model, train_loader, val_loader)

    def predict_image(img, model):
        xb = to_device(img.unsqueeze(0), device)
        yb = model(xb)
        _, preds = torch.max(yb, dim=1)
        return preds[0].item()

    acc = int((evaluate(model, test_loader).get("val_acc")) * 100)
    print(f"\nAfter final validation: {acc}% model accuracy.\n")

    print(f"Additional {print_tests} random tests for user visualization:")
    for x in range(print_tests):
        img, label = test_dataset[random.randint(0, len(test_dataset))]

        print(
            "Label:",
            dataset.classes[label],
            ", Predicted:",
            dataset.classes[predict_image(img, model)],
        )
