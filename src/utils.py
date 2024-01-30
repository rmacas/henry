import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import irfft
from pathlib import Path
from torch.utils.data import DataLoader, random_split


def chdir(dir_, logger, create=False):
    """Directory check; optionally create the directory."""
    if not dir_.is_dir():
        if create:
            logger.warning(f"The directory {dir_} doesn't exist. Creating it")
            dir_.mkdir(parents=True, exist_ok=True)
        else:
            logger.error(f"The directory {dir_} doesn't exist")
            raise SystemExit(1)


def get_layers(width: int, latent_dim: int, act_fn: object, encoder=True):
    """
    Get encoder/decoder layers given latent dim and start/end width. Next
    layer changes by factor 2.
    """
    sizes = []
    size = width
    while size > 30:
        sizes.append(int(size))
        size = size / 2
    sizes.append(latent_dim)

    if encoder is False:
        sizes = sizes[::-1]  # decoder is reversed encoder
    layers = []
    for layer_idx in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[layer_idx], sizes[layer_idx+1]))
        layers.append(act_fn())
    layers.pop()  # remove last act fn as it is not needed
    return layers


class Encoder(nn.Module):
    def __init__(self, width: int, latent_dim: int, act_fn: object = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(*get_layers(width, latent_dim, act_fn))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, width: int, latent_dim: int, act_fn: object = nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(*get_layers(width, latent_dim, act_fn,
                                             encoder=False))

    def forward(self, x):
        return self.net(x)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        width: int,
        latent_dim: int,
        lr: float,
        act_fn: object = nn.ReLU,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
    ):
        super(AutoEncoder, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.width = width
        self.latent_dim = latent_dim
        self.encoder = encoder_class(width, latent_dim, act_fn)
        self.decoder = decoder_class(width, latent_dim, act_fn)
        self.val_loss = []

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, x):
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        loss = self._get_reconstruction_loss(batch)
        self.val_loss.append(loss)
        self.log("val_loss", loss, sync_dist=True)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.val_loss.clear()

    def _test_plot(self, noisy: torch.Tensor, denoised: torch.Tensor, idx: int,
                   loss: float):
        noisy = noisy.cpu().numpy()
        denoised = denoised.cpu().numpy()
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(noisy[idx, :], label='Noisy')
        plt.plot(denoised[idx, :], label='Denoised')
        plt.legend(loc='upper right')
        plt.xlim([0, self.width-1])
        plt.ylabel('Normalized amplitude')
        plt.xlabel('index')
        plt.subplot(1, 2, 2)
        plt.plot(noisy[idx, :] - denoised[idx, :], label='Residual')
        plt.legend()
        plt.xlim([0, self.width-1])
        plt.xlabel('index')
        plt.suptitle(f'Loss: {loss:.7f}')
        plt.savefig(f"reports/figures/"
                    f"{self.width}_{self.latent_dim}_{idx}.png", dpi=200)
        plt.close()

    def test_step(self, batch):
        x_hat = self.forward(batch)
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        for idx in range(10):
            self._test_plot(batch, x_hat, idx, loss)


class GetData():
    def __init__(
        self,
        width: int = 128
    ):
        super().__init__()
        self.width = width

    def _gen_blip(self, fs: int, flow: int, fhigh: int, dt_shift: float):
        "Fake blip model."
        freqs = np.arange(1 + fs//2)
        spec = np.zeros(len(freqs))
        logf = np.log(freqs[flow:fhigh])
        spec1 = (logf-logf[0])*(logf[-1]-logf)
        spec[flow:fhigh] = spec1/np.max(spec1)
        spec_shifted = np.exp(-1j*freqs*2*np.pi*dt_shift)*spec
        blip = np.roll(irfft(spec_shifted), fs//2)
        return blip / np.max(blip)

    def fake_blips(self, fs: int = 512):
        "Generate fake blips."
        dt_shifts = np.random.normal(0, 5, 50) / 1000  # in ms
        f_lows = np.linspace(15, 35, 21, dtype=int)
        f_highs = np.linspace(190, 230, 41, dtype=int)

        blips = []
        for dt_shift in dt_shifts:
            for f_low in f_lows:
                for f_high in f_highs:
                    blip = self._gen_blip(fs, f_low, f_high, dt_shift)
                    blips.append(blip)
        blips = np.array(blips)
        blips = blips[:, fs//2-self.width//2:fs//2+self.width//2]
        blips = np.array(blips).astype('float32')
        return blips

    def shift_blips(self, dset: object = np.array):
        "Generate more blips by shiftng them in time."
        shift_idxs = range(-5, 6)
        shifted_blips = []
        for idx in shift_idxs:
            shifted_blips.append(np.roll(dset, idx, axis=1))
        shifted_blips = np.array(shifted_blips)
        shifted_blips = np.vstack(shifted_blips)
        return shifted_blips

    def real_blips(self, ddir: object = Path('data/external')):
        "Load real whitened blips from O3 Livingston data."
        files = ddir.glob('*.npy')
        blips = []
        for blip in files:
            blip_data = np.load(blip)[:, 1]  # ignore time
            blips.append(blip_data)
        blips = np.array(blips)
        length = len(blips.T)
        blips = blips[:, length//2-self.width//2:length//2+self.width//2]
        blips = blips.astype('float32')
        return blips


class GlitchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: str = 'real',
        batch_size: int = 100,
        width: int = 128
    ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.width = width
        self.prepare_data_per_node = False

    def prepare_data(self):
        gd = GetData(self.width)
        if self.data == 'real':
            self.glitch = F.normalize(torch.from_numpy(gd.real_blips()))
        elif self.data == 'fake':
            self.glitch = F.normalize(torch.from_numpy(gd.fake_blips()))
        else:
            raise SystemExit("Works only with 'real' and 'fake' data.")

    def setup(self, stage=None):
        gen = torch.Generator().manual_seed(0)
        self.train, self.val, self.test = random_split(self.glitch,
                                                       [0.89, 0.1, 0.01],
                                                       generator=gen)

    def train_dataloader(self):
        if self.data == 'real':
            # generate more glitches for training by shifting them
            gd = GetData(self.width)
            return DataLoader(gd.shift_blips(self.train),
                              batch_size=self.batch_size, num_workers=4)
        else:
            return DataLoader(self.train, batch_size=self.batch_size,
                              num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=len(self.test), num_workers=4)
