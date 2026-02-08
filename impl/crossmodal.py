import torch
import os
from matplotlib import pyplot as plt

from .model import UMAPMixture, device
from .util import Config, embed_and_recon

def crossmodal_recon(data: list[torch.Tensor], cfg: Config, model: UMAPMixture | None = None) -> list[torch.Tensor]:
    """Perform text-to-image cross-modal reconstruction and visualize results.

    Embeds text features, reconstructs to image pixel space, and saves
    comparison images to results/ directory.

    Args:
        data: List of [text_tensor, image_tensor] for reconstruction.
        cfg: Configuration object with inference hyperparameters.
        model: Trained UMAPMixture model.

    Returns:
        List of reconstructed image tensors.
    """
    recon = embed_and_recon(model, [data[0]], [0], [1], cfg)[0]

    loss = torch.mean((recon - data[1]).pow(2)).item()
    print(f"Reconstruction loss from text to image: {loss:.4f}")

    recon_images = recon.view(-1, 3, 64, 64).clamp(0, 1).cpu()
    orig_images = data[1].view(-1, 3, 64, 64).clamp(0, 1).cpu()

    if not os.path.exists("results"):
        os.makedirs("results")

    for i in range(orig_images.shape[0]):
        _, axes = plt.subplots(2, 1, figsize=(15, 6))

        axes[0].imshow(orig_images[i].permute(1, 2, 0).numpy())
        axes[0].set_title(f"Original image {i+1}")
        axes[0].axis('off')

        axes[1].imshow(recon_images[i].permute(1, 2, 0).numpy())
        axes[1].set_title(f"Reconstructed from text {i+1}")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"results/recon_text_to_image_{i+1}.png")
        plt.close()

    return [recon]
