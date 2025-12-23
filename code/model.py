import torch
from torch import linalg as LA
from torch import sparse as sp
from torch.autograd import functional as AF
from torch.nn import functional as F
from tqdm import tqdm

class UMAPEncoder:
    def __init__(self, k_neighbors: int, out_dim: int, id: int = 0):
        self.k_neighbors = k_neighbors
        self.out_dim = out_dim
        self.id = id
        self.sigmas = None
        self.rhos = None

    def get_sigmas(self, dists: torch.Tensor, min_dists: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
        def diff(sigmas: torch.Tensor) -> torch.Tensor:
            ps = torch.exp(- (dists - min_dists) / sigmas.unsqueeze(1))
            sums = ps.view(-1, self.k_neighbors).sum(dim=1)

            return sums - target

        sigmas = torch.ones(dists.size(0), requires_grad=True)
        target = torch.log2(torch.tensor(self.k_neighbors))

        for _ in range(num_iters):
            vals = diff(sigmas)
            grads = torch.autograd.grad(vals.sum(), sigmas, create_graph=True)[0]

            sigmas = (sigmas - vals / (grads + 1e-6)).clamp(min=1e-6).detach().requires_grad_(True)

        self.sigmas = sigmas.detach()
        return self.sigmas

    def fuzzy_knn_graph(self, inputs: torch.Tensor, mode: str = "fit", query: torch.Tensor | None = None, ref_data: torch.Tensor | None = None, num_iters: int = 10, a: float | None = None, b: float | None = None) -> torch.Tensor:
        if mode not in ["fit", "transform", "invert"]:
            raise ValueError(f"Invalid mode: {mode}")

        N = inputs.size(0)
        Q = query.size(0) if ref_data is not None else N

        rows = torch.arange(Q).repeat_interleave(self.k_neighbors)
        cols = torch.randint(0, N, (Q * self.k_neighbors,))

        if ref_data is None:
            mask = cols != rows
            rows = rows[mask]
            cols = cols[mask]

        edges = rows * N + cols
        mask = torch.unique(edges)
        rows = mask // N
        cols = mask % N

        ones = torch.ones((mask.size(0),))

        # Batch distance computation to save memory
        num_edges = rows.size(0)
        edge_batch_size = 50000
        dists_list = []

        for i in range(0, num_edges, edge_batch_size):
            end = min(i + edge_batch_size, num_edges)
            batch_rows = rows[i:end]
            batch_cols = cols[i:end]

            batch_dists = LA.vector_norm(inputs[batch_rows] - inputs[batch_cols], dim=1) if ref_data is None else LA.vector_norm(query[batch_rows] - inputs[batch_cols], dim=1)

            dists_list.append(batch_dists)

        dists = torch.cat(dists_list)

        for _ in tqdm(range(num_iters), desc=f"Building graph {self.id}"):
            adj = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), ones, (Q, N)).coalesce()
            if ref_data is None:
                adj = (adj + adj.transpose(0, 1)).coalesce()

            # Batch the matrix multiplication to reduce memory usage
            batch_size = 2000 if N > 15000 else 5000
            all_cand_rows = []
            all_cand_cols = []
            all_cand_vals = []

            for start in range(0, Q, batch_size):
                end = min(start + batch_size, Q)

                batch_mask = (adj.indices()[0] >= start) & (adj.indices()[0] < end)
                if batch_mask.sum() == 0:
                    continue

                batch_indices = adj.indices()[:, batch_mask].clone()
                batch_values = adj.values()[batch_mask]
                # Offset row indices to batch
                batch_indices[0] -= start
                batch_adj = torch.sparse_coo_tensor(batch_indices, batch_values, (end - start, N))

                batch_cand = torch.sparse.mm(batch_adj, adj).coalesce() if ref_data is None else torch.sparse.mm(batch_adj, ref_data).coalesce()

                # Offset row indices back to global
                batch_cand_indices = batch_cand.indices()
                batch_cand_indices[0] += start

                all_cand_rows.append(batch_cand_indices[0])
                all_cand_cols.append(batch_cand_indices[1])
                all_cand_vals.append(batch_cand.values())

            candidates = torch.sparse_coo_tensor(
                torch.stack([torch.cat(all_cand_rows), torch.cat(all_cand_cols)]),
                torch.cat(all_cand_vals),
                (Q, N)
            ).coalesce()

            # Cap max candidates to save memory
            if candidates._nnz() > 50000:
                idx = torch.randperm(candidates._nnz())[:50000]
                indices = candidates.indices()[:, idx]
                values = candidates.values()[idx]
                candidates = torch.sparse_coo_tensor(indices, values, candidates.shape).coalesce()

            cand_rows, cand_cols = candidates.indices()
            cand_dists = LA.vector_norm(inputs[cand_rows] - inputs[cand_cols], dim=1) if ref_data is None else LA.vector_norm(query[cand_rows] - inputs[cand_cols], dim=1)

            if ref_data is None:
                mask = cand_rows != cand_cols

            existing_edges = rows * N + cols
            candidate_edges = cand_rows * N + cand_cols
            is_new = ~torch.isin(candidate_edges, existing_edges)
            
            mask = mask & is_new if ref_data is None else is_new
            cand_rows = cand_rows[mask]
            cand_cols = cand_cols[mask]
            cand_dists = cand_dists[mask]

            all_rows = torch.cat([rows, cand_rows], dim=0)
            all_cols = torch.cat([cols, cand_cols], dim=0)
            all_dists = torch.cat([dists, cand_dists], dim=0)

            # Sort by row, then by distance
            idx = torch.argsort(all_rows * (all_dists.max() + 1e-2) + all_dists, stable=True)
            all_rows = all_rows[idx]
            all_cols = all_cols[idx]
            all_dists = all_dists[idx]

            counts = torch.bincount(all_rows, minlength=Q)
            positions = torch.arange(all_rows.size(0)) - torch.repeat_interleave(torch.cat([torch.tensor([0]), counts.cumsum(0)[:-1]]), counts)

            mask = positions < self.k_neighbors
            rows = all_rows[mask]
            cols = all_cols[mask]
            dists = all_dists[mask]

            ones = torch.ones(rows.size(0))

        dists = dists.view(Q, self.k_neighbors)
        if mode != "invert":
            min_dists = dists.min(dim=1).values.unsqueeze(1).repeat(1, self.k_neighbors)
            sigmas = self.get_sigmas(dists, min_dists)
            weights = torch.exp(- (dists - min_dists) / sigmas.unsqueeze(1)).flatten()
            if mode == "fit":
                self.rhos = min_dists[:, 0]  # First column contains the min dist
        else:
            weights = 1.0 / (1.0 + a * dists.pow(2 * b)).flatten()

        adj = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), weights, (Q, N)).coalesce()
        return adj

    @torch.no_grad()
    def embed_all(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = input.size(0)

        deg = input.sum(dim=1).to_dense().clamp(min=1e-6)
        diag = sp.spdiags(deg.pow(-0.5), torch.zeros(1, dtype=torch.long), (input.size(0), input.size(0)))

        normalized = sp.mm(sp.mm(diag, input), diag)
        identity = sp.spdiags(torch.ones(n), torch.zeros(1, dtype=torch.long), (n, n))
        eps = sp.spdiags(torch.full((n,), 1e-6), torch.zeros(1, dtype=torch.long), (n, n))
        pre = identity - normalized + eps

        _, vectors = torch.lobpcg(pre, k=self.out_dim + 1, largest=False)

        return vectors[:, 1:]
    
    @torch.no_grad()
    def embed_query(self, input: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        row_sums = graph.sum(dim=1).to_dense().clamp(min=1e-6)
        indices = graph.indices()
        values = graph.values() / row_sums[indices[0]]
        normalized = torch.sparse_coo_tensor(indices, values, graph.shape)

        return sp.mm(normalized, input)

    def init(self, input: torch.Tensor, mode: str = "fit", query: torch.Tensor | None = None, ref_data: torch.Tensor | None = None, ref_embeds: torch.Tensor | None = None, a: float | None = None, b: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in ["fit", "transform", "invert"]:
            raise ValueError(f"Invalid mode: {mode}")

        graph = self.fuzzy_knn_graph(input, mode, query, ref_data, num_iters=10, a=a, b=b)
        if mode == "fit":
            graph = (graph + graph.transpose(0, 1) - graph * graph.transpose(0, 1)).coalesce()
            embed = self.embed_all(graph)
        elif mode == "transform":
            embed = self.embed_query(ref_embeds, graph)
        else:
            embed = self.embed_query(input, graph)

        return graph, embed

class UMAPMixture:
    def __init__(self, k_neighbors: int, out_dim: int, min_dist: float, num_encoders: int):
        self.k_neighbors = k_neighbors
        self.out_dim = out_dim
        self.min_dist = min_dist
        self.num_encoders = num_encoders

        self.a, self.b = self.get_ab_coeffs(min_dist)

        self.encoders = [UMAPEncoder(k_neighbors, out_dim, id=i) for i in range(num_encoders)]

        self.data = None
        self.graphs = []
        self.embeds = []

    def _umap_attr_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, a: float, b: float, ref: torch.Tensor | None = None) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx] if ref is not None else embeds[j_idx]

        dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)

        loss = torch.log(1 + a * dist.pow(b)).mean()
        return loss

    def _umap_rep_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, a: float, b: float, ref: torch.Tensor | None = None) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx] if ref is not None else embeds[j_idx]

        dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)

        loss = -torch.log(a * dist.pow(b) / (1 + a * dist.pow(b)) + 1e-6).mean() # 1e-6 avoids ln(0)
        return loss

    def _inv_attr_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, a: float, b: float, ref: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx]

        dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)
        weight = 1.0 / (1.0 + a * dist.pow(b))

        loss = (dist / (weight * sigma[j_idx] + 1e-6)).mean()
        return loss

    def _inv_rep_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, ref: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx]

        dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)
        weight = (- (dist - rho[j_idx]).clamp(min=1e-6) / (sigma[j_idx] + 1e-6)).exp()

        loss = -torch.log(1 - weight + 1e-6).mean()
        return loss

    def _infonce_loss(self, embeds_0: torch.Tensor, embeds_1: torch.Tensor, active: torch.Tensor, n_neg: int = 32, temperature: float = 0.5) -> torch.Tensor:
        i_idx, j_idx = active.indices()[0], active.indices()[1]
        num_pairs = i_idx.size(0)
        num_samples = embeds_1.size(0)

        infonce_batch_size = 10000
        losses = []

        # Batch computation to avoid OOM
        for start in range(0, num_pairs, infonce_batch_size):
            end = min(start + infonce_batch_size, num_pairs)

            batch_i_idx = i_idx[start:end]
            batch_j_idx = j_idx[start:end]

            anchors = embeds_0[batch_i_idx]
            positives = embeds_1[batch_j_idx]

            # Compute L2 norm between normalized anchors and samples
            anchors_norm = F.normalize(anchors, dim=1)
            positives_norm = F.normalize(positives, dim=1)
            pos_sim = (anchors_norm * positives_norm).sum(dim=1) / temperature

            batch_size = batch_i_idx.size(0)
            neg_idx = torch.randint(0, num_samples, (batch_size, n_neg))
            negatives = embeds_1[neg_idx]
            negatives_norm = F.normalize(negatives, dim=2)
            neg_sim = (anchors_norm.unsqueeze(1) * negatives_norm).sum(dim=2) / temperature

            # Compute log softmax directly over all samples at once, and then extract value for only the positive sample
            batch_loss = -torch.log_softmax(torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1), dim=1)[:, 0]
            losses.append(batch_loss.mean())

        return torch.stack(losses).mean()

    def _train(self, embeds: list[torch.Tensor], graphs: list[torch.Tensor], epochs: int, num_rep: int, lr: float, alpha: float, batch_size: int, mode: str = "fit", data_indices: list | None = None, desc: str = "Training", save_dir: str | None = None):
        if mode not in ["fit", "transform", "invert"]:
            raise ValueError(f"Invalid mode: {mode}")

        for rep in embeds:
            rep.requires_grad = True

        if mode == "transform":
            for ref in self.embeds:
                ref.requires_grad = False

        # Adam parameters
        m = [torch.zeros_like(embed) for embed in embeds]
        v = [torch.zeros_like(embed) for embed in embeds]
        beta1, beta2 = 0.9, 0.999
        eps = 1e-6

        loss_history = {
            'total_loss': [],
            'umap_losses': [[] for _ in range(len(embeds))],
            'infonce_losses': [[] for _ in range(len(embeds))] if mode == "fit" else None
        }

        for epoch in tqdm(range(epochs), desc=desc):
            umap_losses = []
            n_modes = len(embeds) if data_indices is None else len(data_indices)

            # Compute UMAP loss for each modality
            for i in range(n_modes):
                embed = embeds[i]
                graph = graphs[i]
                ref_embed = None
                if mode == "transform":
                    ref_embed = self.embeds[data_indices[i]] if data_indices is not None else self.embeds[i]
                elif mode == "invert":
                    ref_embed = self.data[data_indices[i]] if data_indices is not None else self.data[i]

                count = embed.size(0) if mode == "fit" else ref_embed.size(0)
                batch_losses = []

                for j in range(0, count, batch_size):
                    end = min(j + batch_size, count)

                    indices, values = graph.indices(), graph.values()

                    batch = (indices[0] >= j) & (indices[0] < end)
                    batch_indices = indices[:, batch]
                    batch_values = values[batch]

                    keep = torch.rand(batch_values.size(0)) < batch_values
                    i_idx_attr = batch_indices[0][keep]
                    j_idx_attr = batch_indices[1][keep]

                    if mode == "invert":
                        loss_attr = self._inv_attr_loss(embed, i_idx_attr, j_idx_attr, self.a, self.b, ref_embed, self.encoders[i].sigmas)
                    else:
                        loss_attr = self._umap_attr_loss(embed, i_idx_attr, j_idx_attr, self.a, self.b, ref_embed)

                    num_pairs = i_idx_attr.size(0)
                    i_idx_rep = i_idx_attr.repeat_interleave(num_rep)
                    l_idx_rep = torch.randint(0, count, (num_pairs, num_rep)).flatten()

                    if mode == "invert":
                        loss_rep = self._inv_rep_loss(embed, i_idx_rep, l_idx_rep, ref_embed, self.encoders[i].sigmas, self.encoders[i].rhos)
                    else:
                        loss_rep = self._umap_rep_loss(embed, i_idx_rep, l_idx_rep, self.a, self.b, ref_embed)

                    batch_losses.append(loss_attr + loss_rep)

                umap_loss = torch.stack(batch_losses).mean()
                umap_losses.append(umap_loss)

            loss = sum(umap_losses)

            # Compute InfoNCE losses between modalities for cross-modal alignment
            if mode == "fit":
                num_embeds = len(embeds)
                infonce_losses = [torch.tensor(0.0, requires_grad=True) for _ in range(num_embeds)]

                for i in range(num_embeds):
                    for j in range(i + 1, num_embeds):
                        active = graphs[i]

                        infonce_loss = self._infonce_loss(embeds[i], embeds[j], active)

                        infonce_losses[i] = infonce_losses[i] + alpha * infonce_loss
                        infonce_losses[j] = infonce_losses[j] + alpha * infonce_loss

                loss += sum(infonce_losses)

            if save_dir is not None:
                loss_history['total_loss'].append(loss.item())
                for i, umap_loss in enumerate(umap_losses):
                    loss_history['umap_losses'][i].append(umap_loss.item())
                if mode == "fit":
                    for i, infonce_loss in enumerate(infonce_losses):
                        loss_history['infonce_losses'][i].append(infonce_loss.item())

            loss.backward()

            # Adam optimization
            for i in range(len(embeds)):
                grad = embeds[i].grad

                m[i] = beta1 * m[i] + (1 - beta1) * grad
                v[i] = beta2 * v[i] + (1 - beta2) * grad.pow(2)

                m_hat = m[i] / (1 - beta1 ** (epoch + 1))
                v_hat = v[i] / (1 - beta2 ** (epoch + 1))

                embeds[i].data -= lr * m_hat / (v_hat.sqrt() + eps)
                embeds[i].grad.zero_()

        if save_dir is not None:
            import numpy as np
            from pathlib import Path

            Path(save_dir).parent.mkdir(parents=True, exist_ok=True)

            save_dict = {
                'total_loss': np.array(loss_history['total_loss']),
                'epochs': np.arange(len(loss_history['total_loss']))
            }

            for i in range(len(embeds)):
                save_dict[f'umap_loss_{i}'] = np.array(loss_history['umap_losses'][i])

            if mode == "fit" and loss_history['infonce_losses'] is not None:
                for i in range(len(embeds)):
                    save_dict[f'infonce_loss_{i}'] = np.array(loss_history['infonce_losses'][i])

            np.savez(save_dir, **save_dict)

    def fit(self, inputs: list[torch.Tensor], epochs: int, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512, save_dir: str | None = None) -> torch.Tensor | None:
        graphs, embeds = self.init(inputs, mode="fit")
        self.graphs = graphs
        self.data = inputs

        self._train(
            embeds,
            graphs,
            epochs,
            num_rep,
            lr,
            alpha,
            batch_size,
            mode="fit",
            desc=f"Training {self.num_encoders} encoders",
            save_dir=save_dir
        )

        self.embeds = embeds

    def fit_transform(self, inputs: list[torch.Tensor], epochs: int, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        self.fit(inputs, epochs, num_rep, lr, alpha, batch_size)
        return self.embeds

    def transform(self, inputs: list[torch.Tensor], epochs: int, data_indices: list | None = None, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        graphs, embeds = self.init(inputs, mode="transform", data_indices=data_indices)

        self._train(
            embeds,
            graphs,
            epochs,
            num_rep,
            lr,
            alpha,
            batch_size,
            mode="transform",
            data_indices=data_indices,
            desc=f"Embedding {len(embeds)} modalities"
        )

        return embeds
    
    def inverse_transform(self, inputs: list[torch.Tensor], epochs: int, data_indices: list | None = None, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        graphs, embeds = self.init(inputs, mode="invert", data_indices=data_indices)

        self._train(
            embeds,
            graphs,
            epochs,
            num_rep,
            lr,
            alpha,
            batch_size,
            mode="invert",
            data_indices=data_indices,
            desc=f"Inverting {len(embeds)} modalities"
        )

        return embeds

    def get_ab_coeffs(self, min_dist: float, num_iters: int = 50) -> tuple[float, float]:
        def target(dist: torch.Tensor) -> torch.Tensor:
            return torch.where(dist <= min_dist, torch.tensor(1.0), torch.exp(-(dist - min_dist)))

        def estimate(dist: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
            a, b = betas[0].abs() + 1e-6, betas[1].abs() + 1e-6
            return 1.0 / (1.0 + a * dist.pow(2 * b))

        def residuals(distances: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
            return target(distances) - estimate(distances, betas)

        betas = torch.tensor([1.0, 1.0])
        distances = torch.linspace(1e-4, 3.0, 200)

        for _ in tqdm(range(num_iters), desc="Estimating a/b coefficients"):
            res = residuals(distances, betas)
            jac = AF.jacobian(lambda betas: residuals(distances, betas), betas)

            betas -= LA.pinv(jac) @ res

        return (betas[0].abs() + 1e-6).item(), (betas[1].abs() + 1e-6).item()

    def init(self, inputs: list[torch.Tensor], mode: str = "fit", data_indices: list | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if mode not in ["fit", "transform", "invert"]:
            raise ValueError(f"Invalid mode: {mode}")

        graphs = []
        embeds = []

        encoder_indices = data_indices if data_indices is not None else range(self.num_encoders)

        for idx, i in enumerate(tqdm(encoder_indices, desc="Initializing encoders", total=len(encoder_indices))):
            encoder = self.encoders[i]
            if mode == "fit":
                graph, embed = encoder.init(inputs[idx], mode="fit")
            elif mode == "transform":
                graph, embed = encoder.init(self.data[i], mode="transform", query=inputs[idx], ref_data=self.graphs[i], ref_embeds=self.embeds[i])
            else:
                graph, embed = encoder.init(self.data[i], mode="invert", query=inputs[idx], ref_data=self.graphs[i], a=self.a, b=self.b)
            graphs.append(graph)
            embeds.append(embed)

        return graphs, embeds