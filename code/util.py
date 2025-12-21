import torch
from dataclasses import dataclass

from model import UMAPMixture

@dataclass
class Config:
    k_neighbors: int = 15
    out_dim: int = 128
    min_dist: float = 0.1

    train_epochs: int = 100
    num_rep: int = 8
    lr: float = 0.2
    alpha: float = 0.5
    batch_size: int = 512

    test_epochs: int = 20

def get_index(mode: str) -> int:
    mode_to_index = {
        "brain": 0,
        "textual": 1,
        "visual": 2
    }

    return mode_to_index[mode]

def train(data: list[torch.Tensor], cfg: Config) -> UMAPMixture:
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
        batch_size=cfg.batch_size
    )

    return model

def embed(model: UMAPMixture, data: torch.Tensor, src: int, cfg: Config) -> torch.Tensor:
    embed = model.transform(
        data,
        epochs=cfg.test_epochs,
        data_indices=src,
        num_rep=cfg.num_rep,
        lr=cfg.lr,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size
    )

    return embed

def recon(model: UMAPMixture, embed: torch.Tensor, dst: int, cfg: Config) -> torch.Tensor:
    recon = model.inverse_transform(
        embed,
        epochs=cfg.test_epochs,
        data_indices=dst,
        num_rep=cfg.num_rep,
        lr=cfg.lr,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size
    )

    return recon

def embed_and_recon(model: UMAPMixture, data: torch.Tensor, src: int, dst: int, cfg: Config) -> torch.Tensor:
    embed = embed(model, data, src, cfg)
    return recon(model, embed, dst, cfg)
