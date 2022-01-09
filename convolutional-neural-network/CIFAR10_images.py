import os
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import tarfile
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
            "Usage: python CIFAR10_images.py <NUMBER OF EPOCHS> <NUMBER_OF_PRINT_TESTS>"
        )
        sys.exit()

    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, ".")

    if not os.path.isdir("./data"):
        with tarfile.open("./cifar10.tgz", "r:gz") as tar:
            tar.extractall(path="./data")

    data_dir = "./data/cifar10"

    dataset = ImageFolder(data_dir + "/train", transform=ToTensor())

    random_seed = 42
    torch.manual_seed(random_seed)
    val_size = 5000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 128

    train_dl = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

    def apply_kernel(image, kernel):
        ri, ci = image.shape  # image dimensions
        rk, ck = kernel.shape  # kernel dimensions
        ro, co = ri - rk + 1, ci - ck + 1  # output dimensions
        output = torch.zeros([ro, co])
        for i in range(ro):
            for j in range(co):
                output[i, j] = torch.sum(image[i : i + rk, j : j + ck] * kernel)
        return output

    simple_model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2)
    )

    class ImageClassificationBase(nn.Module):
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
            return {"val_loss": loss.detach(), "val_acc": acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x["val_loss"] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
            batch_accs = [x["val_acc"] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
            return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

        def epoch_end(self, epoch, result):
            print(
                "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch, result["train_loss"], result["val_loss"], result["val_acc"]
                )
            )

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class Cifar10CnnModel(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, xb):
            return self.network(xb)

    model = Cifar10CnnModel()

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

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

    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)

    torch.no_grad()

    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(model, val_loader)
            result["train_loss"] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    model = to_device(Cifar10CnnModel(), device)

    opt_func = torch.optim.Adam
    lr = 0.001

    print(f"Training the model over {epochs} epochs:")
    history = fit(epochs, lr, model, train_dl, val_dl, opt_func)

    test_dataset = ImageFolder(data_dir + "/test", transform=ToTensor())

    def predict_image(img, model):
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        return dataset.classes[preds[0].item()]

    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size * 2), device)
    acc = int((evaluate(model, test_loader).get("val_acc")) * 100)
    print(f"\nAfter final tests: {acc}% model accuracy.\n")

    print(f"Additional {print_tests} random tests for user visualization:")
    for x in range(print_tests):
        img, label = test_dataset[random.randint(0, len(test_dataset))]

        print(
            "Label:",
            dataset.classes[label],
            ", Predicted:",
            predict_image(img, model),
        )
