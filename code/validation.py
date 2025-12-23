import torch

from model import UMAPMixture
from util import Config, train, embed
from dataset import load_data, train_test_split

def similarity_test(data: list[torch.Tensor], modes: list[str], cfg: Config, data_dir: str, model: UMAPMixture | None = None, return_values: bool = False) -> float | None:
    if len(modes) == 3:
        embeds = embed(model, data, [0, 1, 2], cfg)
        brain_embeds, text_embeds, image_embeds = embeds[0], embeds[1], embeds[2]

        v1 = brain_embeds - text_embeds
        v2 = brain_embeds - image_embeds

        dists = 0.5 * (torch.norm(v1, dim=1, p=2).pow(2) * torch.norm(v2, dim=1, p=2).pow(2) - torch.dot(v1, v2, dim=1).pow(2)).sqrt()
    elif len(modes) == 2:
        embeds = embed(model, data, [0, 1], cfg)
        embeds_0, embeds_1 = embeds[0], embeds[1]

        dists = torch.norm(embeds_0 - embeds_1, dim=1, p=2)

    else:
        raise ValueError("At least two modes are required for similarity test.")

    result = dists.mean().item()
    print(f"Average cross-modal distance: {result:.4f}")

    if return_values:
        return result

def knn_test(data: list[torch.Tensor], subject: int, modes: list[str], cfg: Config, data_dir: str, k: int = 5, model: UMAPMixture | None = None, return_values: bool = False) -> float | None:
    accs = []
    for (src, dst) in [(i, j) for i in range(len(modes)) for j in range(i+1, len(modes))]:
        embeds = embed(model, [data[src], data[dst]], [src, dst], cfg)
        src_embed, dst_embed = embeds[0], embeds[1]

        sample_indices = torch.arange(src_embed.shape[0]).unsqueeze(1)

        distances = torch.cdist(src_embed, dst_embed, p=2)
        knn_indices_fwd = torch.topk(distances, k, dim=1, largest=False).indices
        knn_indices_bwd = torch.topk(distances.T, k, dim=1, largest=False).indices

        # Convert to int so that 1+1=2 works properly
        matches = (knn_indices_fwd == sample_indices).any(dim=1).int() + (knn_indices_bwd == sample_indices).any(dim=1).int()
        correct = matches.sum().item()

        acc = correct / (2 * src_embed.shape[0])
        accs.append(acc)

    result = torch.tensor(accs).mean().item()
    print(f"Average KNN accuracy: {result:.4f}")

    if return_values:
        return result