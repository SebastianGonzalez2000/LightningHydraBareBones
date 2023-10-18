import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss

class LitMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = os.getcwd(), batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True, transform=ToTensor())
        MNIST(self.data_dir, train=False, download=True, transform=ToTensor())

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=ToTensor())
            self.mnist_train, self.mnist_val = utils.data.random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=ToTensor())

    def train_dataloader(self):
        return utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)


# define the LightningModule
class LitModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        loss = self.model(x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        val_loss = self.model(x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        test_loss = self.model(x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
autoencoder = AutoEncoder(encoder, decoder)

# init the litModule
litModule = LitModule(autoencoder)

# setup data
mnistDataModule = LitMNISTDataModule()

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(
    limit_train_batches=100, 
    limit_val_batches=20,
    limit_test_batches=20,
    max_epochs=1)

trainer.fit(litModule, datamodule=mnistDataModule)

# test the model
trainer.test(model=litModule, datamodule=mnistDataModule)

# load checkpoint
checkpoint = "./lightning_logs/version_21/checkpoints/epoch=0-step=100.ckpt"
litModule = LitModule.load_from_checkpoint(checkpoint, model=autoencoder)

# choose your trained nn.Module
encoder = litModule.model.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=litModule.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)