import torch
from matplotlib import pyplot as plt

from model import UMAPMixture
from util import Config, train, embed, recon
from data import load_labels, load_data, train_test_split

def do_latent_math(dataset: str, subject: int, cfg: Config, classes: tuple[int, int, int], modes: list[str], model: UMAPMixture | None = None) -> list[torch.Tensor]:
    modes = ["brain", "textual", "visual"] if "all" in modes else modes

    train_split = []
    test_split = []
    for mode in modes:
        data = load_data(dataset, mode, subject)
        train_data, test_data = train_test_split(data)

        train_split.append(train_data)
        test_split.append(test_data)

    if model is None:
        model = train(train_split, cfg)

    labels = load_labels(dataset, subject)
    indices = {cls: (labels == cls).nonzero(as_tuple=True)[0] for cls in classes}

    embeds = []
    for cls in classes:
        mode_embeds = []
        for mode_idx, mode in enumerate(modes):
            embed_cls = embed(model, test_split[mode_idx][indices[cls]], mode_idx, cfg)
            mode_embeds.append(torch.mean(embed_cls, dim=0))

        embeds.append(torch.stack(mode_embeds).mean(dim=0))

    result_embeds = embeds[0] - embeds[1] + embeds[2]

    recons = []
    for mode_idx, mode in enumerate(modes):
        recon_mode = recon(model, result_embeds, mode_idx, cfg)
        recons.append(recon_mode)

    return recons

def save_latent_math(dataset: str, subject: int, cfg: Config, classes: tuple[int, int, int], modes: list[str], model: UMAPMixture | None = None) -> None:
    recons = do_latent_math(dataset, subject, cfg, classes, modes, model=model)

    for recon, mode in zip(recons, modes):
        data = load_data(dataset, mode, subject)
        _, test_data = train_test_split(data)

        num_display = 4
        _, axes = plt.subplots(2, num_display, figsize=(15, 6))
        for i in range(num_display):
            axes[0, i].plot(test_data[i].cpu().numpy())
            axes[0, i].set_title(f"Original {mode} sample {i+1}")
            axes[1, i].plot(recon[i].cpu().numpy())
            axes[1, i].set_title(f"Reconstructed {mode} from latent math sample {i+1}")

        plt.tight_layout()
        plt.savefig(f"results/{dataset}_sub-{subject}_{mode}_latent_math_{classes[0]}-{classes[1]}+{classes[2]}.png")
        plt.close()