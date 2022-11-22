import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.datasets.utils import download_url
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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path="./data")

    data_dir = "./data/cifar10"
    classes = os.listdir(data_dir + "/train")

    # Data transforms (normalization & data augmentation)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfms = tt.Compose(
        [
            tt.RandomCrop(32, padding=4, padding_mode="reflect"),
            tt.RandomHorizontalFlip(),
            # tt.RandomRotate
            # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
            # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            tt.ToTensor(),
            tt.Normalize(*stats, inplace=True),
        ]
    )
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    train_ds = ImageFolder(data_dir + "/train", train_tfms)
    valid_ds = ImageFolder(data_dir + "/test", valid_tfms)

    batch_size = 400

    train_dl = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True
    )
    valid_dl = DataLoader(valid_ds, batch_size * 2, num_workers=3, pin_memory=True)

    # def denormalize(images, means, stds):
    #     means = torch.tensor(means).reshape(1, 3, 1, 1)
    #     stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    #     return images * stds + means

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
    valid_dl = DeviceDataLoader(valid_dl, device)

    class SimpleResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            )
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            )
            self.relu2 = nn.ReLU()

        def forward(self, x):
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.conv2(out)
            return (
                self.relu2(out) + x
            )  # ReLU can be applied before or after adding the input

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

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
                "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch,
                    result["lrs"][-1],
                    result["train_loss"],
                    result["val_loss"],
                    result["val_acc"],
                )
            )

    def conv_block(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    class ResNet9(ImageClassificationBase):
        def __init__(self, in_channels, num_classes):
            super().__init__()

            self.conv1 = conv_block(in_channels, 64)
            self.conv2 = conv_block(64, 128, pool=True)
            self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

            self.conv3 = conv_block(128, 256, pool=True)
            self.conv4 = conv_block(256, 512, pool=True)
            self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

            self.classifier = nn.Sequential(
                nn.MaxPool2d(4),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )

        def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out

    model = to_device(ResNet9(3, 10), device)

    torch.no_grad()

    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def fit_one_cycle(
        epochs,
        max_lr,
        model,
        train_loader,
        val_loader,
        weight_decay=0,
        grad_clip=None,
        opt_func=torch.optim.SGD,
    ):
        torch.cuda.empty_cache()
        history = []

        # Set up cutom optimizer with weight decay
        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )

        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(get_lr(optimizer))
                sched.step()

            # Validation phase
            result = evaluate(model, val_loader)
            result["train_loss"] = torch.stack(train_losses).mean().item()
            result["lrs"] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    history = [evaluate(model, valid_dl)]

    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    print(f"Training the model over {epochs} epochs:")
    history += fit_one_cycle(
        epochs,
        max_lr,
        model,
        train_dl,
        valid_dl,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
        opt_func=opt_func,
    )

    def predict_image(img, model):
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        return train_ds.classes[preds[0].item()]

    acc = int((evaluate(model, valid_dl).get("val_acc")) * 100)
    print(f"\nFinal accuracy: {acc}%.\n")

    print(f"Additional {print_tests} random tests for user visualization:")
    for x in range(print_tests):
        img, label = valid_ds[random.randint(0, len(valid_ds))]
        print(
            "Label:", train_ds.classes[label], ", Predicted:", predict_image(img, model)
        )
