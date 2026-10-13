import numpy as np
import torch
import torch.nn as nn

class vEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(vEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            # nn.LayerNorm(normalized_shape=hidden_dim1),
            # nn.InstanceNorm1d(num_features=hidden_dim1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            # nn.LayerNorm(normalized_shape=output_dim),
            # nn.InstanceNorm1d(num_features=output_dim),
            nn.ReLU()
        )

        self.f_mu = nn.Linear(hidden_dim, latent_dim)
        self.f_var = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x):
        return self.f_mu(self.model(x)), self.f_var(self.model(x))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class vDecoder(nn.Module):

    def __init__(self, output_dim, latent_dim, hidden_dim=512):
        super(vDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            # nn.LayerNorm(normalized_shape=hidden_dim1),
            # nn.InstanceNorm1d(num_features=hidden_dim1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            # nn.LayerNorm(normalized_shape=x_dim),
            # nn.InstanceNorm1d(num_features=input_dim),
        )

    def forward(self, z):
        return self.model(z)


