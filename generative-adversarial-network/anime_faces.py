import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.utils import save_image
from PIL import Image
import cv2
import os
from os import listdir
import sys

DATA_DIR = "./data"

if __name__ == "__main__":

    try:

        # Get epochs from function call argument
        epochs = int(sys.argv[1])
        check_corrupted_files = sys.argv[2]

    except Exception as e:

        print(f"Exception: {e}")
        print(
            "Usage: python anime_faces.py <NUMBER OF EPOCHS> <CHECK_CORRUPTED_IMAGES [true/false]>"
        )
        sys.exit()

    def detect_and_delete_corrupted_images():
        for filename in listdir(f"{DATA_DIR}/images"):
            if filename.endswith(".jpg"):
                try:
                    img = Image.open(f"{DATA_DIR}/images/" + filename)
                    img.verify()
                except (IOError, SyntaxError) as e:
                    print(f"Bad file: {filename}, deleting.")
                    os.remove(f"{DATA_DIR}/images/{filename}")

    if check_corrupted_files.upper() == "TRUE":
        print("Detecting and deleting corrupted images in dataset.\n")
        detect_and_delete_corrupted_images()

    image_size = 64
    batch_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    train_ds = ImageFolder(
        DATA_DIR,
        transform=T.Compose(
            [
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(*stats),
            ]
        ),
    )

    train_dl = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True
    )

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

    discriminator = nn.Sequential(
        # in: 3 x 64 x 64
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x 32 x 32
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x 16 x 16
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x 8 x 8
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 512 x 4 x 4
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
        # out: 1 x 1 x 1
        nn.Flatten(),
        nn.Sigmoid(),
    )

    discriminator = to_device(discriminator, device)

    latent_size = 128

    generator = nn.Sequential(
        # in: latent_size x 1 x 1
        nn.ConvTranspose2d(
            latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False
        ),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        # out: 512 x 4 x 4
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        # out: 256 x 8 x 8
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        # out: 128 x 16 x 16
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        # out: 64 x 32 x 32
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
        # out: 3 x 64 x 64
    )

    def denorm(img_tensors):
        return img_tensors * stats[1][0] + stats[0][0]

    xb = torch.randn(batch_size, latent_size, 1, 1)  # random latent tensors

    generator = to_device(generator, device)

    def train_discriminator(real_images, opt_d):
        # Clear discriminator gradients
        opt_d.zero_grad()

        # Pass real images through discriminator
        real_preds = discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        # Generate fake images
        latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
        fake_preds = discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(opt_g):
        # Clear generator gradients
        opt_g.zero_grad()

        # Generate fake images
        latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(latent)

        # Try to fool the discriminator
        preds = discriminator(fake_images)
        targets = torch.ones(batch_size, 1, device=device)
        loss = F.binary_cross_entropy(preds, targets)

        # Update generator weights
        loss.backward()
        opt_g.step()

        return loss.item()

    sample_dir = "generated"
    os.makedirs(sample_dir, exist_ok=True)

    def save_samples(index, latent_tensors):
        fake_images = generator(latent_tensors)
        fake_fname = "generated-images-{0:0=4d}.png".format(index)
        save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
        print("Saving", fake_fname)

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

    print("Saving initial generated images (random noise):")
    save_samples(0, fixed_latent)

    def fit(epochs, lr, start_idx=1):
        torch.cuda.empty_cache()

        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        # Create optimizers
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(epochs):
            for real_images, _ in train_dl:
                # Train discriminator
                loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
                # Train generator
                loss_g = train_generator(opt_g)

            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            # Log losses & scores (last batch)
            print(
                "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                    epoch + 1, epochs, loss_g, loss_d, real_score, fake_score
                )
            )

            # Save generated images
            save_samples(epoch + start_idx, fixed_latent)

        return losses_g, losses_d, real_scores, fake_scores

    lr = 0.0002

    print(f"\nTraining the model over {epochs} epochs:")
    history = fit(epochs, lr)

    vid_fname = "gans_training.avi"

    files = [
        os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if "generated" in f
    ]
    files.sort()

    out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*"MP4V"), 1, (530, 530))
    [out.write(cv2.imread(fname)) for fname in files]
    out.release()
