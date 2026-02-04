import torch
from torch.nn import functional as F

from model import UMAPMixture
from util import Config, embed

def similarity_test(data: list[torch.Tensor], cfg: Config, model: UMAPMixture | None = None, return_values: bool = False) -> float | None:
    num_modes = len(data)
    if num_modes < 2:
        raise ValueError("At least two modes are required for similarity test.")

    embeds = embed(model, data, list(range(num_modes)), cfg)
    embeds = [F.normalize(e, p=2, dim=1) for e in embeds]

    sims_list = []
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            sims_list.append((embeds[i] * embeds[j]).sum(dim=1))
    sims = torch.stack(sims_list, dim=1).mean(dim=1)

    result = sims.mean().item()
    print(f"Average cross-modal cosine similarity: {result:.4f}")

    if return_values:
        return result

def knn_test(data: list[torch.Tensor], cfg: Config, k: int = 5, model: UMAPMixture | None = None, return_values: bool = False) -> float | None:
    accs = []
    for (src, dst) in [(i, j) for i in range(len(data)) for j in range(i+1, len(data))]:
        embeds = embed(model, [data[src], data[dst]], [src, dst], cfg)
        src_embed, dst_embed = embeds[0], embeds[1]

        correct = 0
        for idx in range(src_embed.shape[0]):
            distances_fwd = torch.norm(dst_embed - src_embed[idx], dim=1)
            knns_fwd = torch.topk(distances_fwd, k, largest=False).indices
            if idx in knns_fwd:
                correct += 1

            distances_bwd = torch.norm(src_embed - dst_embed[idx], dim=1)
            knns_bwd = torch.topk(distances_bwd, k, largest=False).indices
            if idx in knns_bwd:
                correct += 1

        acc = correct / (2 * src_embed.shape[0])
        accs.append(acc)

    result = torch.tensor(accs).mean().item()
    print(f"Average KNN accuracy: {result:.4f}")

    if return_values:
        return result