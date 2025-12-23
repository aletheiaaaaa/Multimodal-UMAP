import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from pathlib import Path
from matplotlib import pyplot as plt
import argparse

from util import Config, get_index
from dataset import load_data, train_test_split, load_labels

class PCABaseline:
    def __init__(self, out_dim: int, num_modalities: int, alpha: float = 1.0):
        self.out_dim = out_dim
        self.num_modalities = num_modalities
        self.alpha = alpha

        self.pcas = [PCA(n_components=out_dim) for _ in range(num_modalities)]
        self.aligners = {}

    def fit(self, data: list[torch.Tensor]):
        embeds = []

        for i, (pca, modality_data) in enumerate(zip(self.pcas, data)):
            X = modality_data.numpy()
            pca.fit(X)
            embed = pca.transform(X)
            embeds.append(embed)

        if self.alpha > 0:
            for i in range(self.num_modalities):
                for j in range(i + 1, self.num_modalities):
                    ridge = Ridge(alpha=1.0 / (self.alpha + 1e-6))
                    ridge.fit(embeds[i], embeds[j])
                    self.aligners[(i, j)] = ridge

                    ridge_inv = Ridge(alpha=1.0 / (self.alpha + 1e-6))
                    ridge_inv.fit(embeds[j], embeds[i])
                    self.aligners[(j, i)] = ridge_inv

        self.embeds = [torch.tensor(e, dtype=torch.float32) for e in embeds]

    def transform(self, data: list[torch.Tensor], data_indices: list[int]) -> list[torch.Tensor]:
        embeds = []

        for idx, modality_data in zip(data_indices, data):
            X = modality_data.numpy()
            embed = self.pcas[idx].transform(X)
            embeds.append(torch.tensor(embed, dtype=torch.float32))

        return embeds

    def inverse_transform(self, embeddings: list[torch.Tensor], data_indices: list[int]) -> list[torch.Tensor]:
        recons = []

        for idx, embed in zip(data_indices, embeddings):
            X = embed.numpy()
            recon = self.pcas[idx].inverse_transform(X)
            recons.append(torch.tensor(recon, dtype=torch.float32))

        return recons

def do_crossmodal(model: PCABaseline, data: torch.Tensor, paths: list[tuple[str, str]]) -> torch.Tensor:
    data = data.numpy()
    recons = []

    for (src, dst) in paths:
        src_idx = get_index(src)
        dst_idx = get_index(dst)

        embed = model.pcas[src_idx].transform(data)
        if (src_idx, dst_idx) in model.aligners:
            aligner = model.aligners[(src_idx, dst_idx)]
            embed = aligner.predict(embed)

        recon = model.pcas[dst_idx].inverse_transform(embed)
        recons.append(torch.tensor(recon, dtype=torch.float32))

    return recons

def save_crossmodal(test_data: list[torch.Tensor], subject: int, paths: list[tuple[str, str]], model: PCABaseline) -> None:
    recons = do_crossmodal(model, test_data, paths)

    for (recon, (src, dst)) in zip(recons, paths):
        num_display = 4
        _, axes = plt.subplots(2, num_display, figsize=(15, 6))
        for i in range(num_display):
            axes[0, i].plot(data[get_index(dst)][i].cpu().numpy())
            axes[0, i].set_title(f"Original {dst} sample {i+1}")
            axes[1, i].plot(recon[i].cpu().numpy())
            axes[1, i].set_title(f"Reconstructed {dst} from {src} sample {i+1}")

        plt.tight_layout()
        plt.savefig(f"results/baseline/sub-{subject}_{src}_to_{dst}.png")
        plt.close()

def train(data: list[torch.Tensor], cfg: Config) -> PCABaseline:
    model = PCABaseline(
        out_dim=cfg.out_dim,
        num_modalities=len(data),
        alpha=cfg.alpha
    )

    model.fit(data)

    return model

def embed(model: PCABaseline, data: list[torch.Tensor], src: list[int]) -> list[torch.Tensor]:
    return model.transform(data, src)

def similarity_test(model: PCABaseline, test_data: list[torch.Tensor], modes: list[str]) -> float:
    embeddings = model.transform(test_data, list(range(len(test_data))))

    if len(modes) == 3:
        brain_embeds, text_embeds, image_embeds = embeddings[0], embeddings[1], embeddings[2]
        v1 = brain_embeds - text_embeds
        v2 = brain_embeds - image_embeds
        dists = 0.5 * (torch.norm(v1, dim=1, p=2).pow(2) * torch.norm(v2, dim=1, p=2).pow(2) - torch.sum(v1 * v2, dim=1).pow(2)).sqrt()
    elif len(modes) == 2:
        embeds_0, embeds_1 = embeddings[0], embeddings[1]
        dists = torch.norm(embeds_0 - embeds_1, dim=1, p=2)
    else:
        raise ValueError("At least two modes are required for similarity test.")

    result = dists.mean().item()
    print(f"Baseline average cross-modal distance: {result:.4f}")
    return result

def knn_test(model: PCABaseline, test_data: list[torch.Tensor], modes: list[str], k: int = 10) -> float:
    embeddings = model.transform(test_data, list(range(len(test_data))))

    accs = []
    for (src, dst) in [(i, j) for i in range(len(modes)) for j in range(i+1, len(modes))]:
        src_embed, dst_embed = embeddings[src], embeddings[dst]

        sample_indices = torch.arange(src_embed.shape[0]).unsqueeze(1)

        distances = torch.cdist(src_embed, dst_embed, p=2)
        knn_indices_fwd = torch.topk(distances, k, dim=1, largest=False).indices
        knn_indices_bwd = torch.topk(distances.T, k, dim=1, largest=False).indices

        matches = (knn_indices_fwd == sample_indices).any(dim=1).int() + (knn_indices_bwd == sample_indices).any(dim=1).int()
        correct = matches.sum().item()

        acc = correct / (2 * src_embed.shape[0])
        accs.append(acc)

    result = torch.tensor(accs).mean().item()
    print(f"Baseline average KNN accuracy: {result:.4f}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA Baseline for Cross-modal Learning")
    parser.add_argument("--out_dim", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regression alpha")
    parser.add_argument("--data_dir", type=str, default="~/Desktop/uni/ml_coursework/data", help="Path to data directory")
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument("--modes", type=str, nargs="+", default=["brain", "visual"], choices=["brain", "textual", "visual"], help="Data modalities to use")
    parser.add_argument("--k_test", type=int, default=10, help="Number of neighbors for k-NN test")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use")
    parser.add_argument("--paths", type=str, nargs="+", default=[], help="List of source-destination mode pairs for cross-modal reconstruction")
    parser.add_argument("--classes", type=int, nargs="+", default=[], help="Classes for latent space arithmetic")

    args = parser.parse_args()

    cfg = Config(
        k_neighbors=15,
        out_dim=args.out_dim,
        min_dist=0.1,
        train_epochs=0,
        num_rep=0,
        lr=0.0,
        alpha=args.alpha,
        batch_size=0,
        test_epochs=0
    )

    train_split = []
    test_split = []
    for mode in args.modes:
        data = load_data(mode, args.subject, args.data_dir)
        train_data, test_data = train_test_split(data)
        train_split.append(train_data)
        test_split.append(test_data)

    train_split = [d[:args.num_samples] for d in train_split]
    test_split = [d[:int(args.num_samples * 0.25)] for d in test_split]

    model = train(train_split, cfg)

    sim_score = similarity_test(model, test_split, args.modes)
    knn_score = knn_test(model, test_split, args.modes, k=args.k_test)

    if args.paths:
        path_tuples = [tuple(path.split("_to_")) for path in args.paths]
        save_crossmodal(test_split, args.subject, path_tuples, model)