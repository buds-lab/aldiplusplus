import torch
from torch import nn
from torch.utils.data import DataLoader


class VAE(nn.Module):
    def __init__(self, num_input, latent_dim, hidden_size=[300, 200, 100]):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_input = num_input

        self.encoder = nn.Sequential(
            nn.Linear(num_input, hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.Tanh(),
            nn.Linear(hidden_size[2], latent_dim),
            nn.Tanh(),
        )
        
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var= nn.Linear(latent_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size[2]),
            nn.Tanh(),
            nn.Linear(hidden_size[2], hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], num_input),
            nn.Tanh(),
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick for backprop"""
        if self.training:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return eps*std + mu
        return mu
    
    def encode(self, x):
        """Transform input into latent dimension"""
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        log_var = self.log_var(hidden)
        return mu, log_var

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, x_hat, x, mu, log_var, beta=1):
        kl_loss = 0.5 * torch.sum(torch.exp(log_var) - log_var - 1 + mu**2)
        mse = nn.MSELoss() # reconstruction loss
        recon_loss = mse(x_hat, x)
        
        return recon_loss + beta * kl_loss
