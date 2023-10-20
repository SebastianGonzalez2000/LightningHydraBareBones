import torch
from torch import nn
from torchvision.datasets import MNIST
import lightning.pytorch as pl

import hydra

from models.lit_module import LitModule

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    print(cfg)

    datamodule = hydra.utils.instantiate(cfg.data)
    module = hydra.utils.instantiate(cfg.module)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(module, datamodule=datamodule)

    # test the model
    trainer.test(model=module, datamodule=datamodule)

    
    # load checkpoint
    checkpoint = "./lightning_logs/version_21/checkpoints/epoch=0-step=100.ckpt"
    module = LitModule.load_from_checkpoint(checkpoint, model=module.model)

    # choose your trained nn.Module
    encoder = module.model.encoder
    encoder.eval()

    # embed 4 fake images!
    fake_image_batch = torch.rand(4, 28 * 28, device=module.device)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
    

if __name__ == "__main__":
    main()