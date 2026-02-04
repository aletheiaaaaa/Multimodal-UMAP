import torch
from dataclasses import dataclass

from .model import UMAPMixture

@dataclass
class Config:
    k_neighbors: int
    out_dim: int
    min_dist: float

    train_epochs: int
    num_rep: int
    lr: float
    alpha: float
    batch_size: int

    test_epochs: int

def get_index(mode: str) -> int:
    mode_to_index = {
        "brain": 0,
        "textual": 1,
        "visual": 2
    }

    return mode_to_index[mode]

def train(data: list[torch.Tensor], cfg: Config, save_dir: str | None = None) -> UMAPMixture:
    model = UMAPMixture(
        k_neighbors=cfg.k_neighbors,
        out_dim=cfg.out_dim,
        min_dist=cfg.min_dist,
        num_encoders=len(data)
    )

    model.fit(
        data,
        epochs=cfg.train_epochs,
        num_rep=cfg.num_rep,
        lr=cfg.lr,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        save_dir=save_dir
    )

    return model

def embed(model: UMAPMixture, data: list[torch.Tensor], src: list[int], cfg: Config) -> list[torch.Tensor]:
    data = [d.unsqueeze(0) if d.dim() == 1 else d for d in data]

    embeds = model.transform(
        data,
        epochs=cfg.test_epochs,
        data_indices=src,
        num_rep=cfg.num_rep,
        lr=cfg.lr,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size
    )

    return embeds

def recon(model: UMAPMixture, embeds: list[torch.Tensor], dst: list[int], cfg: Config) -> list[torch.Tensor]:
    embeds = [e.unsqueeze(0) if e.dim() == 1 else e for e in embeds]

    recons = model.inverse_transform(
        embeds,
        epochs=cfg.test_epochs,
        data_indices=dst,
        num_rep=cfg.num_rep,
        lr=cfg.lr,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size
    )

    return recons

def embed_and_recon(model: UMAPMixture, data: list[torch.Tensor], src: list[int], dst: list[int], cfg: Config) -> list[torch.Tensor]:
    embeds = embed(model, data, src, cfg)
    return recon(model, embeds, dst, cfg)
