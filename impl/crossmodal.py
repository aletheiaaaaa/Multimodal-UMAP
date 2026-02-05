import torch 
from matplotlib import pyplot as plt
from diffusers import AutoencoderKL

from .model import UMAPMixture, device
from .util import Config, embed_and_recon

def crossmodal_recon(data: list[torch.Tensor], cfg: Config, model: UMAPMixture | None = None) -> list[torch.Tensor]:
    recons = []
    losses = []

    recon = embed_and_recon(model, [data[0]], [0], [1], cfg)[0]
    recons.append(recon)

    loss = torch.mean((recon - data[1]).pow(2)).item()
    losses.append(loss)
    print(f"Reconstruction loss from text to image: {loss:.4f}")

    autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    orig_img = []
    recon_img = []

    for i, recon_latent in enumerate(recons):
        recon_latent = recon_latent.view(-1, 4, 32, 32).to(device)
        orig_latent = data[1].view(-1, 4, 32, 32).to(device)

        with torch.no_grad():
            recon_images = autoencoder.decode(recon_latent).sample
            orig_images = autoencoder.decode(orig_latent).sample

        recon_images = (recon_images / 2 + 0.5).clamp(0, 1).cpu()
        orig_images = (orig_images / 2 + 0.5).clamp(0, 1).cpu()

        recon_img.append(recon_images)
        orig_img.append(orig_images)


    for i in range(orig_images.shape[0]):
        _, axes = plt.subplots(2, 1, figsize=(15, 6))

        axes[0, 0].imshow(orig_images[i].permute(1, 2, 0).numpy())
        axes[0, 0].set_title(f"Original image {i+1}")
        axes[0, 0].axis('off')

        axes[1, 0].imshow(recon_images[i].permute(1, 2, 0).numpy())
        axes[1, 0].set_title(f"Reconstructed from text {i+1}")
        axes[1, 0].axis('off')

        plt.tight_layout()
        plt.savefig(f"results/recon_text_to_image_{i+1}.png")
        plt.close()

    return recons