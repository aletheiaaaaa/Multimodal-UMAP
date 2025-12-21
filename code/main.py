import argparse

from validation import triangle_test, knn_test
from crossmodal import save_crossmodal
from arithmetic import save_latent_math
from util import Config, train

def init_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal UMAP Mixture Model Experiments")

    parser.add_argument("--k_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--out_dim", type=int, default=128, help="Output embedding dimension")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance for UMAP")

    parser.add_argument("--train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_rep", type=int, default=8, help="Number of repetitions")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    parser.add_argument("--dataset", type=str, default="DIR-Wiki", choices=["DIR-Wiki", "GOD-Wiki", "ThingsEEG-Text"], help="Dataset to use")
    parser.add_argument("--subject", type=int, default=0, help="Subject number")

    parser.add_argument("--test_epochs", type=int, default=20, help="Number of testing epochs")
    parser.add_argument("--k_test", type=int, default=5, help="Number of neighbors for k-NN test")
    parser.add_argument("--paths", type=str, nargs="+", default=[], help="List of source-destination mode pairs for cross-modal reconstruction")
    parser.add_argument("--classes", type=int, nargs="+", default=[], help="Classes for latent space arithmetic performed using 'idx1-idx2+idx3'")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = init_parser()
    cfg = Config(
        k_neighbors=args.k_neighbors,
        out_dim=args.out_dim,
        min_dist=args.min_dist,
        train_epochs=args.train_epochs,
        num_rep=args.num_rep,
        lr=args.lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        test_epochs=args.test_epochs
    )

    model = train(args.dataset, args.subject, cfg)

    triangle_test(args.dataset, args.subject, cfg, model=model)
    knn_test(args.dataset, args.subject, cfg, k=args.k_test, model=model)

    if args.paths:
        path_tuples = [tuple(path.split("_to_")) for path in args.paths]
        save_crossmodal(args.dataset, args.subject, cfg, path_tuples, model=model)

    if args.classes:
        if len(args.classes) != 3:
            raise ValueError("Please provide exactly three class indices for latent space arithmetic.")
        save_latent_math(args.dataset, args.subject, cfg, tuple(args.classes), modes=["all"], model=model)