import torch 
from matplotlib import pyplot as plt

from model import UMAPMixture
from util import Config, train, embed_and_recon, get_index
from dataset import load_data, train_test_split

def do_crossmodal(data: list[torch.Tensor], cfg: Config, paths: list[tuple[str, str]], model: UMAPMixture | None = None) -> list[torch.Tensor]:
    recons = []
    for (src, dst) in paths:
        src_idx = get_index(src)
        dst_idx = get_index(dst)

        recon = embed_and_recon(model, [data[src_idx]], [src_idx], [dst_idx], cfg)[0]
        recons.append(recon)

    losses = [torch.mean((recon - data[get_index(dst)]).pow(2)).item() for recon, (_, dst) in zip(recons, paths)]

    for (path, loss) in zip(paths, losses):
        print(f"Reconstruction loss from {path[0]} to {path[1]}: {loss:.4f}")

    return recons

def save_crossmodal(data: list[torch.Tensor], subject: int, cfg: Config, paths: list[tuple[str, str]], data_dir: str, model: UMAPMixture | None = None) -> None:
    recons = do_crossmodal(data, cfg, paths, model=model)

    for (recon, (src, dst)) in zip(recons, paths):
        num_display = 4
        _, axes = plt.subplots(2, num_display, figsize=(15, 6))
        for i in range(num_display):
            axes[0, i].plot(data[get_index(dst)][i].cpu().numpy())
            axes[0, i].set_title(f"Original {dst} sample {i+1}")
            axes[1, i].plot(recon[i].cpu().numpy())
            axes[1, i].set_title(f"Reconstructed {dst} from {src} sample {i+1}")

        plt.tight_layout()
        plt.savefig(f"results/sub-{subject}_{src}_to_{dst}.png")
        plt.close()