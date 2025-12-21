import torch

from model import UMAPMixture
from util import Config, train, embed
from data import load_data, train_test_split

def triangle_test(dataset: str, subject: int, cfg: Config, model: UMAPMixture | None = None) -> None:
    train_split = []
    test_split = []

    for mode in ["brain", "textual", "visual"]:
        data = load_data(dataset, mode, subject)
        train_data, test_data = train_test_split(data)

        train_split.append(train_data)
        test_split.append(test_data)

    if model is None:
        model = train(train_split, cfg)

    areas = []
    for (scan, caption, image) in zip(test_split[0], test_split[1], test_split[2]):
        brain_embed = embed(model, scan.unsqueeze(0), 0, cfg)
        text_embed = embed(model, caption.unsqueeze(0), 1, cfg)
        image_embed = embed(model, image.unsqueeze(0), 2, cfg)

        v1 = brain_embed - text_embed
        v2 = brain_embed - image_embed

        area = 0.5 * (torch.norm(v1, p=2).pow(2) * torch.norm(v2, p=2).pow(2) - torch.dot(v1, v2).pow(2)).sqrt()
        areas.append(area)

    print(f"Average triangle area for subject {subject} in dataset {dataset}: {torch.stack(areas).mean():.4f}")

def knn_test(dataset: str, subject: int, cfg: Config, k: int = 5, model: UMAPMixture | None = None) -> None:
    train_split = []
    test_split = []

    for mode in ["brain", "textual", "visual"]:
        data = load_data(dataset, mode, subject)
        train_data, test_data = train_test_split(data)

        train_split.append(train_data)
        test_split.append(test_data)

    if model is None:
        model = train(train_split, cfg)

    accs = []
    for (src, dst) in [(i, j) for i in range(3) for j in range(3) if i != j]:
        src_embed = embed(model, test_split[src], src, cfg)
        dst_embed = embed(model, test_split[dst], dst, cfg)

        correct = 0
        for i in range(src_embed.shape[0]):
            distances = torch.norm(dst_embed - src_embed[i].unsqueeze(0), dim=1)
            knn_indices = torch.topk(distances, k, largest=False).indices

            if i in knn_indices:
                correct += 1

        acc = correct / src_embed.shape[0]
        accs.append(acc)

    print(f"Average KNN accuracy for subject {subject} in dataset {dataset}: {torch.tensor(accs).mean():.4f}")