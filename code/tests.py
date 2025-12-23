from util import Config, train
from dataset import load_data, train_test_split
from validation import similarity_test, knn_test

def evaluate_config(cfg, train_split, test_split, modes, k_test, save_dir=None):
    model = train(train_split, cfg, save_dir=save_dir)

    sim_score = similarity_test(test_split, modes, cfg, data_dir="", model=model, return_values=True)
    knn_score = knn_test(test_split, 1, modes, cfg, data_dir="", k=k_test, model=model, return_values=True)

    return sim_score, knn_score

def test_option(name, vals, cfg, train_split, test_split, modes, k_test, best_params=None, select_best=True):
    best_val = None
    best_score = float('inf')
    results = []

    print(f"\nTESTING {name.upper()}\n")

    for val in vals:
        params = {
            'k_neighbors': cfg.k_neighbors,
            'out_dim': cfg.out_dim,
            'min_dist': cfg.min_dist,
            'train_epochs': cfg.train_epochs,
            'num_rep': cfg.num_rep,
            'lr': cfg.lr,
            'alpha': cfg.alpha,
            'batch_size': cfg.batch_size,
            'test_epochs': cfg.test_epochs
        }

        if best_params:
            params.update(best_params)

        params[name] = val
        cfg = Config(**params)

        filename = f"losses/{name}_{val}_lr{params['lr']}_alpha{params['alpha']}_numrep{params['num_rep']}_outdim{params['out_dim']}.npz"

        sim_score, knn_score = evaluate_config(cfg, train_split, test_split, modes, k_test, save_dir=filename)
        results.append((val, sim_score, knn_score))

        print(f"{name}={val}: sim={sim_score:.4f}, knn={knn_score:.4f}")

        if select_best and sim_score < best_score:
            best_score = sim_score
            best_val = val

    if select_best:
        print(f"\nBest {name}: {best_val} (sim score: {best_score:.4f})\n")

    return best_val, results

if __name__ == "__main__":
    subject = 1
    modes = ["brain", "visual"]
    data_dir = "~/Desktop/uni/ml_coursework/data"
    k_test = 10
    num_samples = 10000

    base_cfg = Config(
        k_neighbors=15,
        out_dim=128,
        min_dist=0.1,
        train_epochs=200,
        num_rep=8,
        lr=0.2,
        alpha=1.0,
        batch_size=32,
        test_epochs=50
    )

    train_split = []
    test_split = []
    for mode in modes:
        data = load_data(mode, subject, data_dir)
        train_data, test_data = train_test_split(data)
        train_split.append(train_data)
        test_split.append(test_data)

    train_split = [d[:num_samples] for d in train_split]
    test_split = [d[:int(num_samples * 0.25)] for d in test_split]

    best_lr, lr_results = test_option(
        'lr', [0.05, 0.1, 0.2, 0.5], base_cfg, train_split, test_split, modes, k_test
    )

    best_alpha, alpha_results = test_option(
        'alpha', [0.5, 1.0, 2.0, 4.0], base_cfg, train_split, test_split, modes, k_test,
        best_params={'lr': best_lr}
    )

    best_num_rep, num_rep_results = test_option(
        'num_rep', [4, 8, 16], base_cfg, train_split, test_split, modes, k_test,
        best_params={'lr': best_lr, 'alpha': best_alpha}
    )

    _, out_dim_results = test_option(
        'out_dim', [64, 128, 192], base_cfg, train_split, test_split, modes, k_test,
        best_params={'lr': best_lr, 'alpha': best_alpha, 'num_rep': best_num_rep},
        select_best=False
    )

    print("\nFINAL SUMMARY\n")
    print(f"Best LR: {best_lr}")
    print(f"Best alpha: {best_alpha}")
    print(f"Best num_rep: {best_num_rep}")
    print("\nResults by out_dim:")
    for out_dim, sim_score, knn_score in out_dim_results:
        print(f"  out_dim={out_dim}: sim={sim_score:.4f}, knn={knn_score:.4f}")
