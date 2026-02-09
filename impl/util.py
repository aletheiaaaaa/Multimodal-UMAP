import torch
from dataclasses import dataclass
from matplotlib import pyplot as plt

from .model import UMAPMixture

@dataclass
class Config:
    """Configuration for training and inference.

    Attributes:
        k_neighbors: Number of neighbors for kNN graph construction.
        out_dim: Dimensionality of the output embedding space.
        min_dist: Minimum distance parameter for UMAP curve fitting.
        train_epochs: Number of epochs for model training.
        num_rep: Number of negative samples per positive for repulsion loss.
        lr: Learning rate for optimizer.
        alpha: Weight for InfoNCE cross-modal alignment loss.
        batch_size: Batch size for training.
        test_epochs: Number of epochs for transform/inverse_transform.
    """
    k_neighbors: int
    out_dim: int
    min_dist: float

    train_epochs: int
    num_rep: int
    lr: float
    alpha: float
    batch_size: int

    test_epochs: int

def train(data: dict, cfg: Config) -> UMAPMixture:
    """Train a multimodal UMAP model on the provided data.

    Args:
        data: Dictionary mapping modality names to tensors of shape (N, D).
        cfg: Configuration object with training hyperparameters.

    Returns:
        Trained UMAPMixture model.
    """
    data = [data[key] for key in data]

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
    )

    return model

def embed(model: UMAPMixture, data: list[torch.Tensor], src: list[int], cfg: Config) -> list[torch.Tensor]:
    """Embed data into the learned latent space.

    Args:
        model: Trained UMAPMixture model.
        data: List of tensors to embed, one per modality.
        src: Encoder indices specifying which encoder to use for each input.
        cfg: Configuration object with inference hyperparameters.

    Returns:
        List of embedding tensors in the shared latent space.
    """
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
    """Reconstruct data from embeddings back to original feature space.

    Args:
        model: Trained UMAPMixture model.
        embeds: List of embedding tensors to reconstruct.
        dst: Encoder indices specifying target modality for each reconstruction.
        cfg: Configuration object with inference hyperparameters.

    Returns:
        List of reconstructed tensors in the original feature spaces.
    """
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
    """Embed data and reconstruct to a different modality (cross-modal translation).

    Args:
        model: Trained UMAPMixture model.
        data: List of source tensors to translate.
        src: Encoder indices for source modalities.
        dst: Encoder indices for target modalities.
        cfg: Configuration object with inference hyperparameters.

    Returns:
        List of reconstructed tensors in the target modality spaces.
    """
    embeds = embed(model, data, src, cfg)
    return recon(model, embeds, dst, cfg)

def embed_and_plot(model: UMAPMixture, data: list[torch.Tensor], src: list[int], cfg: Config) -> list[torch.Tensor]:
    """Embed data and plot the resulting embeddings in 2D space.

    Args:
        model: Trained UMAPMixture model.
        data: List of tensors to embed and plot.
        src: Encoder indices specifying which encoder to use for each input.
        cfg: Configuration object with inference hyperparameters.

    Returns:
        List of embedding tensors in the shared latent space.
    """

    embeds = embed(model, data, src, cfg)

    if cfg.out_dim != 2:
        print("Output dimension is not 2, skipping plotting.")
        return embeds

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    for i, embed in enumerate(embeds):
        plt.scatter(embed[:, 0].cpu(), embed[:, 1].cpu(), color=colors[i % len(colors)], label=f'Modality {src[i]}', alpha=0.5)
    plt.legend()
    plt.title("UMAP Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

    plt.savefig("images/umap_embeddings.png")

    return embeds
