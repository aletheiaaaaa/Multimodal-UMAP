import torch 
from matplotlib import pyplot as plt

from model import UMAPMixture
from util import Config, embed_and_recon, get_index

def crossmodal_recon(data: list[torch.Tensor], cfg: Config, paths: list[tuple[str, str]], model: UMAPMixture | None = None) -> list[torch.Tensor]:
    recons = []
    losses = []
    for (src, dst) in paths:
        src_idx = get_index(src)
        dst_idx = get_index(dst)

        recon = embed_and_recon(model, [data[src_idx]], [src_idx], [dst_idx], cfg)[0]
        recons.append(recon)

        loss = torch.mean((recon - data[dst_idx]).pow(2)).item()
        losses.append(loss)
        print(f"Reconstruction loss from {src} to {dst}: {loss:.4f}")

    for (recon, (src, dst)) in zip(recons, paths):
        num_display = 4
        _, axes = plt.subplots(2, num_display, figsize=(15, 6))
        for i in range(num_display):
            axes[0, i].plot(data[get_index(dst)][i].cpu().numpy())
            axes[0, i].set_title(f"Original {dst} sample {i+1}")
            axes[1, i].plot(recon[i].cpu().numpy())
            axes[1, i].set_title(f"Reconstructed {dst} from {src} sample {i+1}")

        plt.tight_layout()
        plt.savefig(f"results/recon_{src}_to_{dst}.png")
        plt.close()

    return recons