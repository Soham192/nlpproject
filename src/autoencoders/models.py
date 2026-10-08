"""The four AE variants, all built on one shared Backbone.

Fairness: every variant has the same encoder/decoder layer sizes and latent size.
The only differences are what defines each variant:
  dense   plain MSE reconstruction
  sparse  + L1 penalty on the latent activations
  dae     corrupted input, clean target (corruption in data.py, strength = min_snr_db; network identical to dense)
  vae     encoder emits (mu, logvar) — the extra logvar head is the only parameter difference — plus KL
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import VARIANTS, AEConfig


class Backbone(nn.Module):
    """MLP encoder/decoder. `latent_mult=2` makes the encoder emit mu and logvar for the VAE."""

    def __init__(self, input_dim: int, hidden: list[int], latent: int, latent_mult: int = 1) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        d = input_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc.append(nn.Linear(d, latent * latent_mult))
        dec: list[nn.Module] = []
        d = latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec.append(nn.Linear(d, input_dim))
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)


@dataclass
class AEOutput:
    recon: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None


class AutoEncoder(nn.Module):
    """Dense AE. Subclasses change only the loss / latent, never the layer sizes."""

    variant = "dense"
    corrupts_input = False  # True for the DAE: train.py feeds corrupt(x) and targets x

    def __init__(self, input_dim: int, hidden: list[int], latent: int, latent_mult: int = 1) -> None:
        super().__init__()
        self.input_dim, self.latent = input_dim, latent
        self.net = Backbone(input_dim, hidden, latent, latent_mult)

    def forward(self, x: torch.Tensor) -> AEOutput:
        z = self.net.encoder(x)
        return AEOutput(self.net.decoder(z), z)

    def loss(self, x_in: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        out = self(x_in)
        mse = torch.mean((out.recon - target) ** 2)
        return mse, {"mse": float(mse.detach())}

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window anomaly score (B,): mean squared reconstruction error. Low = looks like training speech."""
        return torch.mean((self(x).recon - x) ** 2, dim=1)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).recon

    def hparams(self) -> dict[str, float]:
        return {}


class SparseAE(AutoEncoder):
    variant = "sparse"

    def __init__(self, *a, l1_weight: float, **kw) -> None:
        super().__init__(*a, **kw)
        self.l1_weight = float(l1_weight)

    def loss(self, x_in, target):
        out = self(x_in)
        mse = torch.mean((out.recon - target) ** 2)
        l1 = torch.mean(torch.abs(out.z))
        return mse + self.l1_weight * l1, {"mse": float(mse.detach()), "l1": float(l1.detach())}

    def hparams(self):
        return {"l1_weight": self.l1_weight}


class DenoisingAE(AutoEncoder):
    """Identical network to the dense AE; only its training input is corrupted. `min_snr_db` sets the
    corruption strength (training SNR ~ U(min_snr_db, upper bound from config))."""

    variant = "dae"
    corrupts_input = True

    def __init__(self, *a, min_snr_db: float, **kw) -> None:
        super().__init__(*a, **kw)
        self.min_snr_db = float(min_snr_db)

    def corruption_cfg(self, base: dict) -> dict:
        lo, hi = base["snr_db"]
        if not self.min_snr_db < hi:
            raise ValueError(f"min_snr_db={self.min_snr_db} must be below the upper SNR bound {hi}")
        return {**base, "snr_db": [self.min_snr_db, hi]}

    def hparams(self):
        return {"min_snr_db": self.min_snr_db}


class VAE(AutoEncoder):
    """Gaussian VAE. Loss per window = (SSE + beta * KL) / D, i.e. MSE + beta * KL / D.

    score() is the negative ELBO with beta = 1, using the posterior mean for reconstruction.
    """

    variant = "vae"

    def __init__(self, input_dim: int, hidden: list[int], latent: int, beta: float) -> None:
        super().__init__(input_dim, hidden, latent, latent_mult=2)
        self.beta = float(beta)

    def _encode(self, x):
        mu, logvar = self.net.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, x):
        mu, logvar = self._encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return AEOutput(self.net.decoder(z), z, mu, logvar)

    @staticmethod
    def _kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)

    def loss(self, x_in, target):
        out = self(x_in)
        sse = torch.sum((out.recon - target) ** 2, dim=1)
        kl = self._kl(out.mu, out.logvar)
        loss = torch.mean(sse + self.beta * kl) / self.input_dim
        return loss, {"mse": float(sse.mean().detach()) / self.input_dim,
                      "kl": float(kl.mean().detach())}

    @torch.no_grad()
    def score(self, x):
        out = self(x)  # eval mode -> z = mu
        sse = torch.sum((out.recon - x) ** 2, dim=1)
        return (sse + self._kl(out.mu, out.logvar)) / self.input_dim

    def hparams(self):
        return {"beta": self.beta}


def build(variant: str, cfg: AEConfig, hparam: float | None = None) -> AutoEncoder:
    """Construct a variant from config. `hparam` is the dev-tuned knob (l1_weight / beta) if the variant has one."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    dims = (cfg.features.input_dim, list(cfg.model["hidden"]), int(cfg.model["latent"]))
    grid = cfg.hparam_grid(variant)
    if grid is not None and hparam is None:
        raise ValueError(f"{variant} needs its {grid[0]} (one of {grid[1]})")
    if variant == "dense":
        return AutoEncoder(*dims)
    if variant == "sparse":
        return SparseAE(*dims, l1_weight=hparam)
    if variant == "dae":
        return DenoisingAE(*dims, min_snr_db=hparam)
    return VAE(*dims, beta=hparam)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
