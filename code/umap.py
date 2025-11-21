import torch
from torch import linalg as LA
from torch import sparse as sp
from torch.autograd.functional import jacobian
from torch.nn import functional as F
from tqdm import tqdm

class UMAPEncoder:
    def __init__(self, k_neighbors: int, out_dim: int, id: int = 0):
        self.k_neighbors = k_neighbors
        self.out_dim = out_dim
        self.id = id
        self.sigmas = None

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

    def fuzzy_knn_graph(self, inputs: torch.Tensor, query: torch.Tensor | None = None, ref: torch.Tensor | None = None, num_iters: int = 10) -> torch.Tensor:
        N = inputs.size(0)
        Q = query.size(0) if ref is not None else N

        rows = torch.arange(Q).repeat_interleave(self.k_neighbors)
        cols = torch.randint(0, N, (Q * self.k_neighbors,))

        # Remove self-intersections and duplicates
        if ref is None:
            mask = cols != rows
            rows = rows[mask]
            cols = cols[mask]

        edges = rows * N + cols
        mask = torch.unique(edges)
        rows = mask // N
        cols = mask % N

        ones = torch.ones((mask.size(0),))

        dists = LA.vector_norm(inputs[rows] - inputs[cols], dim=1) if ref is None else LA.vector_norm(query[rows] - inputs[cols], dim=1)

        for _ in tqdm(range(num_iters), desc=f"Building graph {self.id}"):
            adj = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), ones, (Q, N)).coalesce()
            if ref is None:
                adj = (adj + adj.transpose(0, 1)).coalesce()
    
            candidates = sp.mm(adj, adj).coalesce() if ref is None else sp.mm(adj, ref).coalesce()
            if candidates._nnz() > 1000000:
                idx = torch.randperm(candidates._nnz())[:1000000]
                indices = candidates.indices()[:, idx]
                values = candidates.values()[idx]
                candidates = torch.sparse_coo_tensor(indices, values, candidates.shape).coalesce()

            cand_rows, cand_cols = candidates.indices()
            cand_dists = LA.vector_norm(inputs[cand_rows] - inputs[cand_cols], dim=1) if ref is None else LA.vector_norm(query[cand_rows] - inputs[cand_cols], dim=1)

            # Remove existing edges and self-intersections
            if ref is None:
                mask = cand_rows != cand_cols

            existing_edges = rows * N + cols
            candidate_edges = cand_rows * N + cand_cols
            is_new = ~torch.isin(candidate_edges, existing_edges)
            
            mask = mask & is_new if ref is None else is_new
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
        min_dists = dists.min(dim=1).values.unsqueeze(1).repeat(1, self.k_neighbors)
        sigmas = self.get_sigmas(dists, min_dists)
        weights = torch.exp(- (dists - min_dists) / sigmas.unsqueeze(1)).flatten()

        adj = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), weights, (Q, N))
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

        (_, vectors) = torch.lobpcg(pre, k=self.out_dim + 1, largest=False)

        return vectors[:, 1:]
    
    @torch.no_grad()
    def embed_query(self, orig: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        row_sums = graph.sum(dim=1).to_dense().clamp(min=1e-6)
        indices = graph.indices()
        values = graph.values() / row_sums[indices[0]]
        normalized = torch.sparse_coo_tensor(indices, values, graph.shape)

        return sp.mm(normalized, orig)

    def init(self, input: torch.Tensor, mode: str = "fit", query: torch.Tensor | None = None, ref: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in ["fit", "transform", "invert"]:
            raise ValueError(f"Invalid mode: {mode}")

        graph = self.fuzzy_knn_graph(input, query, ref, num_iters=10)
        if mode == "fit":
            graph = (graph + graph.transpose(0, 1) - graph * graph.transpose(0, 1)).coalesce() # Symmetrize fuzzy adjacency matrix
            embed = self.embed_all(graph)
        else:  # mode == "transform"
            embed = self.embed_query(ref, graph)

        return graph, embed

class UMAPMixture:
    def __init__(self, k_neighbors: int, out_dim: int, min_dist: float, num_encoders: int = 3):
        self.k_neighbors = k_neighbors
        self.out_dim = out_dim
        self.min_dist = min_dist
        self.num_encoders = num_encoders

        self.a, self.b = self.get_ab_coeffs(min_dist)

        self.encoders = [UMAPEncoder(k_neighbors, out_dim, id=i) for i in range(num_encoders)]

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

        loss = dist / (weight * sigma[j_idx] + 1e-6).mean()
        return loss

    def _inv_rep_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, ref: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx]

        dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)
        rho = embeds[j_idx].min(dim=1).values
        weight = (- (dist - rho).clamp(min=1e-6) / (sigma[j_idx] + 1e-6)).exp()

        loss = -torch.log(1 - weight + 1e-6).mean()
        return loss

    def _infonce_loss(self, embeds_0: torch.Tensor, embeds_1: torch.Tensor, active: torch.Tensor, n_neg: int = 32, temperature: float = 0.5) -> torch.Tensor:
        i_idx, j_idx = active.indices()[0], active.indices()[1]

        anchors = embeds_0[i_idx]
        positives = embeds_1[j_idx]

        # Compute L2 norm between normalized anchors and positive samples
        anchors_norm = F.normalize(anchors, dim=1)
        positives_norm = F.normalize(positives, dim=1)
        pos_sim = (anchors_norm * positives_norm).sum(dim=1) / temperature

        num_pairs = i_idx.size(0)
        num_samples = embeds_1.size(0)

        neg_idx = torch.randint(0, num_samples, (num_pairs, n_neg))
        negatives = embeds_1[neg_idx]
        negatives_norm = F.normalize(negatives, dim=2)
        neg_sim = (anchors_norm.unsqueeze(1) * negatives_norm).sum(dim=2) / temperature

        # Compute log softmax directly over all samples at once, and then extract value for only the positive sample
        loss = -torch.log_softmax(torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1), dim=1)[:, 0]

        return loss.mean()

    def _train(self, embeds: list[torch.Tensor], graphs: list[torch.Tensor], epochs: int, num_rep: int, lr: float, alpha: float, batch_size: int, mode: str = "fit", data_indices: list | None = None, desc: str = "Training"):
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

        for epoch in tqdm(range(epochs), desc=desc):
            umap_losses = []

            # Compute UMAP loss for each modality
            for i in range(len(embeds)):
                embed = embeds[i]
                graph = graphs[i]
                ref_embed = None
                if mode == "transform":
                    ref_embed = self.embeds[data_indices[i]] if data_indices is not None else self.embeds[i]

                count = embed.size(0)
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
                        loss_rep = self._inv_rep_loss(embed, i_idx_rep, l_idx_rep, ref_embed, self.encoders[i].sigmas)
                    else:
                        loss_rep = self._umap_rep_loss(embed, i_idx_rep, l_idx_rep, self.a, self.b, ref_embed)

                    batch_losses.append(loss_attr + loss_rep)

                umap_loss = torch.stack(batch_losses).mean()
                umap_losses.append(umap_loss)

            # Compute InfoNCE losses between modalities
            num_embeds = len(embeds)
            infonce_losses = [torch.tensor(0.0, requires_grad=True) for _ in range(num_embeds)]

            if mode != "invert":
                for i in range(num_embeds):
                    for j in range(i + 1, num_embeds):
                        active = graphs[i]

                        if mode == "transform":
                            ref_j = self.embeds[data_indices[j]] if data_indices is not None else self.embeds[j]
                            infonce_loss = self._infonce_loss(embeds[i], ref_j, active)
                        else:
                            infonce_loss = self._infonce_loss(embeds[i], embeds[j], active)

                        infonce_losses[i] = infonce_losses[i] + alpha * infonce_loss
                        infonce_losses[j] = infonce_losses[j] + alpha * infonce_loss

            loss = sum(umap_losses) + sum(infonce_losses)
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

    def fit(self, inputs: list[torch.Tensor], epochs: int, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor | None:
        graphs, embeds = self.init(inputs, mode="fit")
        self.graphs = graphs

        self._train(
            embeds,
            graphs,
            epochs,
            num_rep,
            lr,
            alpha,
            batch_size,
            mode="fit",
            desc=f"Training {self.num_encoders} encoders"
        )

        self.embeds = embeds

    def fit_transform(self, inputs: list[torch.Tensor], epochs: int, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        self.fit(inputs, epochs, num_rep, lr, alpha, batch_size)
        return self.embeds

    def transform(self, inputs: list[torch.Tensor], epochs: int, data_indices: list | None = None, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        graphs, embeds = self.init(inputs, mode="transform")

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
            jac = jacobian(lambda betas: residuals(distances, betas), betas)

            betas -= LA.pinv(jac) @ res

        return (betas[0].abs() + 1e-6).item(), (betas[1].abs() + 1e-6).item()

    def init(self, inputs: list[torch.Tensor], mode: str = "fit") -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if mode not in ["fit", "transform", "invert"]:
            raise ValueError(f"Invalid mode: {mode}")

        graphs = []
        embeds = []

        for i, encoder in tqdm(enumerate(self.encoders), desc="Initializing encoders", total=self.num_encoders):
            if mode == "fit":
                graph, embed = encoder.init(inputs[i], mode="fit")
            elif mode == "transform":
                graph, embed = encoder.init(self.embeds[i], mode="transform", query=inputs[i], ref=self.graphs[i])
            else:
                graph, embed = encoder.init(self.embeds[i], mode="invert", query=inputs[i], ref=self.graphs[i])
            graphs.append(graph)
            embeds.append(embed)

        return graphs, embeds
