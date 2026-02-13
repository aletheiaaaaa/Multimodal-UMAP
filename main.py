import torch
import argparse
import os

from impl.validation import similarity_test, knn_test
from impl.crossmodal import crossmodal_recon
from impl.util import Config, train
from impl.dataset import load_data
from impl.model import UMAPMixture

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

def init_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal UMAP Mixture Model Experiments")

    parser.add_argument("--k_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--out_dim", type=int, default=2, help="Output embedding dimension")
    parser.add_argument("--min_dist", type=float, default=0.01, help="Minimum distance for UMAP")

    parser.add_argument("--train_epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--num_rep", type=int, default=8, help="Number of repulsive points for UMAP")
    parser.add_argument("--train_lr", type=float, default=0.1, help="Training learning rate")
    parser.add_argument("--test_lr", type=float, default=0.001, help="Test/transform learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Cross-modal alignment weight")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to log training losses")

    parser.add_argument("--test_epochs", type=int, default=80, help="Number of testing epochs")
    parser.add_argument("--k_test", type=int, default=1, help="Number of neighbors for k-NN test")
    parser.add_argument("--crossmodal", type=str, default="yes", choices=["yes", "no"], help="Whether to save cross-modal reconstructions")

    parser.add_argument("--load_pretrained", type=str, default="no", choices=["yes", "no"], help="Whether to load a pretrained model")
    parser.add_argument("--save_path", type=str, default="models/flickr30k.pt", help="Path to save the trained model")

    args = parser.parse_args()

    return args

def plot_embeddings(embeds_0, embeds_1, title="Embedding Overlay", path="embed_overlay.png"):
    combined = torch.cat([embeds_0, embeds_1], dim=0).detach().cpu().numpy()
    coords = PCA(n_components=2).fit_transform(combined)
    n = embeds_0.shape[0]
    plt.figure(figsize=(8, 8))
    plt.scatter(coords[n:, 0], coords[n:, 1], s=4, alpha=0.4, label="mod1", c="tab:orange")
    plt.scatter(coords[:n, 0], coords[:n, 1], s=4, alpha=0.4, label="mod0", c="tab:blue")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

def plot_baseline(train_split, path="baseline_overlay.png"):
    texts_np = train_split["texts"].cpu().numpy()
    images_np = train_split["images"].cpu().numpy()
    text_embeds = umap.UMAP(n_components=2).fit_transform(texts_np)
    image_embeds = umap.UMAP(n_components=2).fit_transform(images_np)
    plt.figure(figsize=(8, 8))
    plt.scatter(image_embeds[:, 0], image_embeds[:, 1], s=4, alpha=0.4, label="images", c="tab:orange")
    plt.scatter(text_embeds[:, 0], text_embeds[:, 1], s=4, alpha=0.4, label="texts", c="tab:blue")
    plt.legend()
    plt.title("Baseline: Independent UMAP per modality")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

if __name__ == "__main__":
    args = init_parser()
    cfg = Config(
        k_neighbors=args.k_neighbors,
        out_dim=args.out_dim,
        min_dist=args.min_dist,
        train_epochs=args.train_epochs,
        num_rep=args.num_rep,
        train_lr=args.train_lr,
        test_lr=args.test_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        test_epochs=args.test_epochs
    )

    train_split = load_data(split="train")
    test_split = load_data(split="test")

    plot_baseline(train_split)

    if args.load_pretrained == "yes":
        model = UMAPMixture.load_state_dict(args.save_path)
    else:
        model = train(train_split, cfg)

    if args.save_path is not None:
        model.save_state_dict(args.save_path)

    plot_embeddings(model.embeds[0], model.embeds[1], title="Train", path="train_overlay.png")

    similarity_test(test_split, cfg, model=model)
    knn_test(test_split, cfg, k=args.k_test, model=model)

    if args.crossmodal == "yes":
        indices = torch.randperm(test_split["texts"].shape[0])[:16]
        samples = list(v[indices] for v in test_split.values())
        crossmodal_recon(samples, cfg, model=model)