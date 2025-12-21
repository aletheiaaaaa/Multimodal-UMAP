import torch 
from matplotlib import pyplot as plt

from model import UMAPMixture
from util import Config, train, embed_and_recon, get_index
from data import load_data, train_test_split

def do_crossmodal(dataset: str, subject: int, cfg: Config, paths: list[tuple[str]], model: UMAPMixture | None = None) -> list[torch.Tensor]:
    modes = [set(mode for path in paths for mode in path)]

    train_split = []
    test_split = []
    for mode in modes:
        data = load_data(dataset, mode, subject)
        train_data, test_data = train_test_split(data)

        train_split.append(train_data)
        test_split.append(test_data)

    if model is None:
        model = train(train_split, cfg)

    recons = []
    for (src, dst) in paths:
        src_idx = get_index(src)
        dst_idx = get_index(dst)

        recon = embed_and_recon(model, test_split[src_idx], src_idx, dst_idx, cfg)
        recons.append(recon)

    losses = [torch.mean((recon - test_split[get_index(dst)]).pow(2)).item() for recon, (_, dst) in zip(recons, paths)]

    for (path, loss) in zip(paths, losses):
        print(f"Reconstruction loss from {path[0]} to {path[1]}: {loss:.4f}")

    return recons

def save_crossmodal(dataset: str, subject: int, cfg: Config, paths: list[tuple[str]], model: UMAPMixture | None = None) -> None:
    recons = do_crossmodal(dataset, subject, cfg, paths, model=model)

    for (recon, (src, dst)) in zip(recons, paths):
        data = load_data(dataset, dst, subject)
        _, test_data = train_test_split(data)

        num_display = 4
        _, axes = plt.subplots(2, num_display, figsize=(15, 6))
        for i in range(num_display):
            axes[0, i].plot(test_data[i].cpu().numpy())
            axes[0, i].set_title(f"Original {dst} sample {i+1}")
            axes[1, i].plot(recon[i].cpu().numpy())
            axes[1, i].set_title(f"Reconstructed {dst} from {src} sample {i+1}")

        plt.tight_layout()
        plt.savefig(f"results/{dataset}_sub-{subject}_{src}_to_{dst}.png")
        plt.close()