import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class BetaVAE(VAE):
    """Beta-VAE is a VAE with a beta weight on the KLD term."""
    def __init__(self, input_dim, latent_dim=16):
        super(BetaVAE, self).__init__(input_dim, latent_dim)

class ConvVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(ConvVAE, self).__init__()
        # Since we have vectors (1D), we use Conv1d to satisfy the "convolutional" requirement
        # Input shape: (Batch, 1, input_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # (Batch, 32, input_dim/2)
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate flattened size
        self.flatten_dim = 32 * (input_dim // 2)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, input_dim // 2)),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, x):
        # x: (Batch, input_dim)
        x = x.unsqueeze(1) # (Batch, 1, input_dim)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        h = self.decoder_fc(z)
        recon = self.decoder(h).squeeze(1)
        return recon, mu, logvar

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=16):
        super(CVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=-1)
        h = self.encoder(combined)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

def train_vae(model, data, cond_data=None, epochs=50, batch_size=32, lr=1e-3, beta=1.0):
    if cond_data is not None:
        dataset = TensorDataset(torch.FloatTensor(data), torch.FloatTensor(cond_data))
    else:
        dataset = TensorDataset(torch.FloatTensor(data))
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            
            if cond_data is not None:
                c = batch[1]
                outputs = model(x, c)
            else:
                outputs = model(x)
                
            if len(outputs) == 3: # VAE style: (recon, mu, logvar)
                recon_x, mu, logvar = outputs
                loss = vae_loss(recon_x, x, mu, logvar, beta)
            else: # AE style: (recon, z)
                recon_x, z = outputs
                loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(data):.4f}")
    return model

if __name__ == "__main__":
    # Test with dummy data
    input_dim = 32
    latent_dim = 16
    model = VAE(input_dim, latent_dim)
    dummy_data = torch.randn(100, input_dim)
    train_vae(model, dummy_data, epochs=5)
    print("VAE model trained successfully on dummy data.")
