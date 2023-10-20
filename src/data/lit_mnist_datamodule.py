import os
import torch
from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl


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