from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size, e_l1_out, e_l2_out, d_l1_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, e_l1_out), nn.ReLU(), nn.Linear(e_l1_out, e_l2_out))
        self.decoder = nn.Sequential(nn.Linear(e_l2_out, d_l1_out), nn.ReLU(), nn.Linear(d_l1_out, input_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return loss