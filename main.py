import argparse

from impl.validation import similarity_test, knn_test
# from crossmodal import save_crossmodal
from impl.util import Config, train
from impl.dataset import load_data

def init_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal UMAP Mixture Model Experiments")

    parser.add_argument("--k_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--out_dim", type=int, default=16, help="Output embedding dimension")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance for UMAP")

    parser.add_argument("--train_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--num_rep", type=int, default=8, help="Number of repulsive points for UMAP")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Cross-modal alignment weight")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to log training losses")

    # parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save cross-modal reconstructions")

    parser.add_argument("--test_epochs", type=int, default=100, help="Number of testing epochs")
    parser.add_argument("--k_test", type=int, default=1, help="Number of neighbors for k-NN test")
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

    train_split = load_data(split="train")
    test_split = load_data(split="validation")

    model = train(train_split, cfg, args.save_dir)

    similarity_test(test_split, cfg, model=model)
    knn_test(test_split, cfg, k=args.k_test, model=model)

    # if args.paths:
    #     path_tuples = [tuple(path.split("_to_")) for path in args.paths]
    #     save_crossmodal(test_split, args.subject, cfg, path_tuples, model=model, data_dir=args.data_dir)