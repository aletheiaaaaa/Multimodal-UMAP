import argparse

from validation import similarity_test, knn_test
from crossmodal import save_crossmodal
from util import Config, train
from dataset import load_data, train_test_split

def init_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal UMAP Mixture Model Experiments")

    parser.add_argument("--k_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--out_dim", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance for UMAP")

    parser.add_argument("--train_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--num_rep", type=int, default=8, help="Number of repulsive points for UMAP")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Cross-modal alignment weight")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use for training")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to log training losses")

    parser.add_argument("--data_dir", type=str, default="~/Desktop/uni/ml_coursework/data", help="Path to data directory")
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument("--modes", type=str, nargs="+", default=["brain", "visual"], choices=["brain", "textual", "visual"], help="Data modalities to use")

    parser.add_argument("--test_epochs", type=int, default=50, help="Number of testing epochs")
    parser.add_argument("--k_test", type=int, default=10, help="Number of neighbors for k-NN test")
    parser.add_argument("--paths", type=str, nargs="+", default=[], help="List of source-destination mode pairs for cross-modal reconstruction")

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

    # Validation
    similarity_test(test_split, args.modes, cfg, data_dir=args.data_dir, model=model)
    knn_test(test_split, args.subject, args.modes, cfg, data_dir=args.data_dir, k=args.k_test, model=model)

    # Experiments
    if args.paths:
        path_tuples = [tuple(path.split("_to_")) for path in args.paths]
        save_crossmodal(test_split, args.subject, cfg, path_tuples, model=model, data_dir=args.data_dir)