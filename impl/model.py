import torch
import os
from torch import linalg as LA
from torch import sparse as sp
from torch import autograd
from torch.autograd import functional as AF
from torch.nn import functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UMAPEncoder:
    """Single-modality UMAP encoder for graph construction and spectral embedding.

    Handles fuzzy kNN graph construction using NN-descent and initial spectral
    embedding via normalized Laplacian eigenvectors.

    Attributes:
        k_neighbors: Number of neighbors for kNN graph.
        out_dim: Output embedding dimensionality.
        id: Encoder identifier for progress display.
        sigmas: Learned bandwidth parameters for fuzzy set membership.
        rhos: Distance to nearest neighbor for each point.
    """

    def __init__(self, k_neighbors: int, out_dim: int, id: int = 0):
        self.k_neighbors = k_neighbors
        self.out_dim = out_dim
        self.id = id
        self.sigmas = None
        self.rhos = None

    def get_sigmas(self, dists: torch.Tensor, min_dists: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
        """Compute sigma values for fuzzy set membership using Newton's method.

        Finds sigma such that sum of membership probabilities equals log2(k_neighbors).

        Args:
            dists: Distance matrix of shape (N, k_neighbors).
            min_dists: Minimum distances (rho) for each point.
            num_iters: Number of Newton iterations.

        Returns:
            Tensor of sigma values with shape (N,).
        """
        def diff(sigmas: torch.Tensor) -> torch.Tensor:
            ps = torch.exp(- (dists - min_dists) / sigmas.unsqueeze(1))
            sums = ps.view(-1, self.k_neighbors).sum(dim=1)

            return sums - target

        sigmas = torch.ones(dists.size(0), requires_grad=True).to(device)
        target = torch.log2(torch.tensor(self.k_neighbors)).to(device)

        for _ in range(num_iters):
            vals = diff(sigmas)
            grads = autograd.grad(vals.sum(), sigmas, create_graph=True)[0]

            sigmas = (sigmas - vals / (grads + 1e-6)).clamp(min=1e-6).detach().requires_grad_(True)

        self.sigmas = sigmas.detach()
        return self.sigmas

    def fuzzy_knn_graph(self, inputs: torch.Tensor, mode: str = "fit", query: torch.Tensor | None = None, ref_data: torch.Tensor | None = None, num_iters: int = 10, a: float | None = None, b: float | None = None) -> torch.Tensor:
        """Build approximate kNN graph using NN-descent algorithm.

        Constructs a fuzzy simplicial set representation of the data manifold
        with memory-efficient batched distance computation.

        Args:
            inputs: Input data tensor of shape (N, D).
            mode: One of "fit", "transform", or "invert".
            query: Query points for transform/invert modes.
            ref_data: Reference adjacency for neighbor candidates.
            num_iters: Number of NN-descent iterations.
            a: UMAP curve parameter (for invert mode).
            b: UMAP curve parameter (for invert mode).

        Returns:
            Sparse adjacency tensor with fuzzy membership weights.
        """
        N = inputs.size(0)
        Q = query.size(0) if query is not None else N

        rows = torch.arange(Q).repeat_interleave(self.k_neighbors).to(device)
        cols = torch.randint(0, N, (Q * self.k_neighbors,)).to(device)

        if ref_data is None:
            mask = cols != rows
            rows = rows[mask]
            cols = cols[mask]

        edges = rows * N + cols
        mask = torch.unique(edges)
        rows = mask // N
        cols = mask % N

        ones = torch.ones((mask.size(0),)).to(device)

        # Batch distance computation to save memory
        num_edges = rows.size(0)
        edge_batch_size = 2000 if N > 15000 else 20000
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
            positions = (torch.arange(all_rows.size(0), device=device) - torch.repeat_interleave(torch.cat([torch.tensor([0], device=device), counts.cumsum(0)[:-1]]), counts))

            mask = positions < self.k_neighbors
            rows = all_rows[mask]
            cols = all_cols[mask]
            dists = all_dists[mask]

            ones = torch.ones(rows.size(0)).to(device)

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
        """Compute spectral embedding via LOBPCG on normalized Laplacian.

        Args:
            input: Sparse adjacency matrix.

        Returns:
            Eigenvectors corresponding to smallest non-trivial eigenvalues.
        """
        n = input.size(0)

        deg = input.sum(dim=1).to_dense().clamp(min=1e-6)
        offsets = torch.zeros(1, dtype=torch.long)
        diag = sp.spdiags(deg.cpu().pow(-0.5), offsets, (input.size(0), input.size(0))).to(device)

        normalized = sp.mm(sp.mm(diag, input), diag)
        identity = sp.spdiags(torch.ones(n), offsets, (n, n)).to(device)
        eps = sp.spdiags(torch.full((n,), 1e-6), offsets, (n, n)).to(device)
        pre = identity - normalized + eps

        _, vectors = torch.lobpcg(pre, k=self.out_dim + 1, largest=False)

        return vectors[:, 1:]
    
    @torch.no_grad()
    def embed_query(self, ref: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Embed query points using weighted average of reference embeddings.

        Args:
            ref: Reference embeddings tensor.
            query: Sparse affinity matrix from query to reference points.

        Returns:
            Query embeddings as weighted combination of reference embeddings.
        """
        row_sums = query.sum(dim=1).to_dense().clamp(min=1e-6)
        indices = query.indices()
        values = query.values() / row_sums[indices[0]]
        normalized = torch.sparse_coo_tensor(indices, values, query.shape).coalesce()

        return sp.mm(normalized, ref)

    def init(self, input: torch.Tensor, mode: str = "fit", query: torch.Tensor | None = None, ref_data: torch.Tensor | None = None, ref_embeds: torch.Tensor | None = None, a: float | None = None, b: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize graph and embeddings for a single modality.

        Args:
            input: Input data or reference data depending on mode.
            mode: "fit" for training, "transform" for embedding, "invert" for reconstruction.
            query: Query points for transform/invert modes.
            ref_data: Reference graph for neighbor lookup.
            ref_embeds: Reference embeddings for out-of-sample extension.
            a: UMAP curve parameter.
            b: UMAP curve parameter.

        Returns:
            Tuple of (adjacency graph, initial embeddings).
        """
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
    """Multimodal UMAP model with cross-modal alignment.

    Learns a shared embedding space for multiple modalities using UMAP's
    manifold learning combined with InfoNCE contrastive alignment.

    Attributes:
        k_neighbors: Number of neighbors for kNN graphs.
        out_dim: Shared embedding dimensionality.
        min_dist: Minimum distance for UMAP curve.
        num_encoders: Number of modality-specific encoders.
        a, b: Fitted UMAP curve parameters.
        encoders: List of UMAPEncoder instances.
        data: Training data for each modality.
        graphs: Fuzzy kNN graphs for each modality.
        embeds: Learned embeddings for each modality.
    """

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
            return torch.tensor(0.0, requires_grad=True).to(device)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx] if ref is not None else embeds[j_idx]

        dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)

        loss = torch.log(1 + a * dist.pow(b)).mean()
        return loss

    def _umap_rep_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, a: float, b: float, ref: torch.Tensor | None = None) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True).to(device)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx] if ref is not None else embeds[j_idx]

        dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)

        loss = -torch.log(a * dist.pow(b) / (1 + a * dist.pow(b)) + 1e-6).mean() # 1e-6 avoids ln(0)
        return loss

    def _inv_attr_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, a: float, b: float, ref: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True).to(device)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx]

        sq_dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)
        dist = sq_dist.sqrt()
        weight = 1.0 / (1.0 + a * sq_dist.pow(b))

        loss = (dist / (weight * sigma[j_idx] + 1e-6)).mean()
        return loss

    def _inv_rep_loss(self, embeds: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, ref: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        if i_idx.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True).to(device)

        embeds_i = embeds[i_idx]
        embeds_j = ref[j_idx]

        sq_dist = ((embeds_i - embeds_j).pow(2)).sum(dim=1).clamp(min=1e-6)
        dist = sq_dist.sqrt()
        weight = (- (dist - rho[j_idx]).clamp(min=1e-6) / (sigma[j_idx] + 1e-6)).exp()

        loss = -torch.log(1 - weight + 1e-6).mean()
        return loss

    def _infonce_loss(self, embeds_0: torch.Tensor, embeds_1: torch.Tensor, n_neg: int = 8, temperature: float = 0.5) -> torch.Tensor:
        num_samples = min(embeds_0.size(0), embeds_1.size(0))
        if num_samples == 0:
            return torch.tensor(0.0, requires_grad=True).to(device)

        batch_size = 1000
        losses = []

        # Batch computation to avoid OOM
        indices = torch.randperm(num_samples).to(device)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            anchors_norm = F.normalize(embeds_0[batch_indices], dim=1)
            positives_norm = F.normalize(embeds_1[batch_indices], dim=1)
            pos_sim = (anchors_norm * positives_norm).sum(dim=1) / temperature

            true_batch_size = batch_indices.size(0)
            neg_idx = torch.randint(0, num_samples, (true_batch_size, n_neg + 1)).to(device)
            negatives = embeds_1[neg_idx]

            mask = neg_idx != batch_indices.unsqueeze(1)
            negatives = negatives * mask.unsqueeze(2)
            negatives_norm = F.normalize(negatives, dim=2)
            neg_sim = ((anchors_norm.unsqueeze(1) * negatives_norm).sum(dim=2) / temperature).masked_fill(~mask, float('-inf'))

            batch_loss = -torch.log_softmax(torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1), dim=1)[:, 0]
            losses.append(batch_loss.mean())

        return torch.stack(losses).mean()

    def _train(self, embeds: list[torch.Tensor], graphs: list[torch.Tensor], epochs: int, num_rep: int, lr: float, alpha: float, batch_size: int, mode: str = "fit", data_indices: list | None = None, desc: str = "Training"):
        embeds = [e.clone().detach().to(device).requires_grad_(True) for e in embeds]

        if mode == "transform":
            for ref in self.embeds:
                ref.requires_grad = False

        optimizer = torch.optim.Adam(embeds, lr=lr)

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

                count = embed.size(0) if mode == "fit" else max(embed.size(0), ref_embed.size(0))
                batch_losses = []

                for j in range(0, count, batch_size):
                    end = min(j + batch_size, count)

                    indices, values = graph.indices(), graph.values()

                    batch = (indices[0] >= j) & (indices[0] < end)
                    batch_indices = indices[:, batch]
                    batch_values = values[batch]

                    keep = torch.rand(batch_values.size(0)).to(device) < batch_values
                    i_idx_attr = batch_indices[0][keep]
                    j_idx_attr = batch_indices[1][keep]

                    if mode == "invert":
                        loss_attr = self._inv_attr_loss(embed, i_idx_attr, j_idx_attr, self.a, self.b, ref_embed, self.encoders[i].sigmas)
                    else:
                        loss_attr = self._umap_attr_loss(embed, i_idx_attr, j_idx_attr, self.a, self.b, ref_embed)

                    num_pairs = i_idx_attr.size(0)
                    i_idx_rep = i_idx_attr.repeat_interleave(num_rep)
                    l_idx_rep = torch.randint(0, count, (num_pairs, num_rep)).flatten().to(device)

                    if mode == "invert":
                        loss_rep = self._inv_rep_loss(embed, i_idx_rep, l_idx_rep, ref_embed, self.encoders[i].sigmas, self.encoders[i].rhos)
                    else:
                        loss_rep = self._umap_rep_loss(embed, i_idx_rep, l_idx_rep, self.a, self.b, ref_embed)

                    batch_losses.append(loss_attr + loss_rep)

                umap_loss = torch.stack(batch_losses).mean()
                umap_losses.append(umap_loss)

            loss = sum(umap_losses)

            # Compute InfoNCE losses between modalities for cross-modal alignment
            if mode != "invert":
                num_embeds = len(embeds)
                infonce_losses = [torch.tensor(0.0, requires_grad=True).to(device) for _ in range(num_embeds)]

                for i in range(num_embeds):
                    for j in range(i + 1, num_embeds):
                        infonce_ij = self._infonce_loss(embeds[i], embeds[j])
                        infonce_ji = self._infonce_loss(embeds[j], embeds[i])
                        infonce_loss = (infonce_ij + infonce_ji) / 2.0

                        infonce_losses[i] = infonce_losses[i] + alpha * infonce_loss
                        infonce_losses[j] = infonce_losses[j] + alpha * infonce_loss

                loss += sum(infonce_losses)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 50 == 0 or epoch == epochs - 1:
                tqdm.write(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

        return embeds

    def fit(self, inputs: list[torch.Tensor], epochs: int, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor | None:
        """Fit the model to multimodal training data.

        Args:
            inputs: List of tensors, one per modality with shape (N, D_i).
            epochs: Number of training epochs.
            num_rep: Negative samples per positive.
            lr: Learning rate.
            alpha: InfoNCE loss weight.
            batch_size: Batch size.
        """
        graphs, embeds = self.init(inputs, mode="fit")
        self.graphs = graphs
        self.data = [x.to(device) for x in inputs]

        self.embeds = self._train(
            embeds,
            graphs,
            epochs,
            num_rep,
            lr,
            alpha,
            batch_size,
            mode="fit",
            desc=f"Training {self.num_encoders} encoders",
        )

    def fit_transform(self, inputs: list[torch.Tensor], epochs: int, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        """Fit the model and return training embeddings.

        Args:
            inputs: List of tensors, one per modality.
            epochs: Number of training epochs.
            num_rep: Negative samples per positive.
            lr: Learning rate.
            alpha: InfoNCE loss weight.
            batch_size: Batch size.

        Returns:
            List of embedding tensors for training data.
        """
        self.fit(inputs, epochs, num_rep, lr, alpha, batch_size)
        return self.embeds

    def transform(self, inputs: list[torch.Tensor], epochs: int, data_indices: list | None = None, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        """Embed new data into the learned latent space.

        Args:
            inputs: List of tensors to embed.
            epochs: Number of optimization epochs.
            data_indices: Which encoder to use for each input.
            num_rep: Negative samples per positive.
            lr: Learning rate.
            alpha: InfoNCE loss weight.
            batch_size: Batch size.

        Returns:
            List of embedding tensors.
        """
        graphs, embeds = self.init(inputs, mode="transform", data_indices=data_indices)

        return self._train(
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
    
    def inverse_transform(self, inputs: list[torch.Tensor], epochs: int, data_indices: list | None = None, num_rep: int = 8, lr: float = 0.2, alpha: float = 0.5, batch_size: int = 512) -> torch.Tensor:
        """Reconstruct original features from embeddings.

        Args:
            inputs: List of embedding tensors to invert.
            epochs: Number of optimization epochs.
            data_indices: Target modality for each reconstruction.
            num_rep: Negative samples per positive.
            lr: Learning rate.
            alpha: Loss weight (unused in invert mode).
            batch_size: Batch size.

        Returns:
            List of reconstructed feature tensors.
        """
        graphs, embeds = self.init(inputs, mode="invert", data_indices=data_indices)

        return self._train(
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

    def get_ab_coeffs(self, min_dist: float, num_iters: int = 50) -> tuple[float, float]:
        """Fit UMAP curve parameters a and b using Gauss-Newton optimization.

        Finds a, b such that 1/(1 + a*d^(2b)) approximates the target membership function.

        Args:
            min_dist: Minimum distance parameter controlling tightness.
            num_iters: Number of Gauss-Newton iterations.

        Returns:
            Tuple of (a, b) coefficients.
        """
        def target(dist: torch.Tensor) -> torch.Tensor:
            return torch.where(dist <= min_dist, torch.tensor(1.0).to(device), torch.exp(-(dist - min_dist)))

        def estimate(dist: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
            a, b = betas[0].abs() + 1e-6, betas[1].abs() + 1e-6
            return 1.0 / (1.0 + a * dist.pow(2 * b))

        def residuals(distances: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
            return target(distances) - estimate(distances, betas)

        betas = torch.tensor([1.0, 1.0]).to(device)
        distances = torch.linspace(1e-4, 3.0, 200).to(device)

        for _ in tqdm(range(num_iters), desc="Estimating a/b coefficients"):
            res = residuals(distances, betas)
            jac = AF.jacobian(lambda betas: residuals(distances, betas), betas)

            betas -= LA.pinv(jac) @ res

        return (betas[0].abs() + 1e-6).item(), (betas[1].abs() + 1e-6).item()

    def init(self, inputs: list[torch.Tensor], mode: str = "fit", data_indices: list | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initialize graphs and embeddings for all modalities.

        Args:
            inputs: Input tensors for each modality.
            mode: "fit", "transform", or "invert".
            data_indices: Encoder indices for transform/invert modes.

        Returns:
            Tuple of (graphs list, embeddings list).
        """
        if mode not in ["fit", "transform", "invert"]:
            raise ValueError(f"Invalid mode: {mode}")

        inputs = [x.to(device) for x in inputs]
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

    def save_state_dict(self, path: str) -> None:
        """Save complete model state to disk.

        Warning: This includes training data, graphs, and embeddings.

        Args:
            path: File path for the checkpoint.
        """
        print(f"Warning: save_state_dict() saves the entire model state, which includes the source dataset. Make sure this is intended before proceeding.")

        state_dict = {
            'k_neighbors': self.k_neighbors,
            'out_dim': self.out_dim,
            'min_dist': self.min_dist,
            'num_encoders': self.num_encoders,
            'a': self.a,
            'b': self.b,
            'encoders': [{
                'sigmas': encoder.sigmas,
                'rhos': encoder.rhos
            } for encoder in self.encoders],
            'data': self.data,
            'graphs': self.graphs,
            'embeds': self.embeds
        }

        torch.save(state_dict, path)

    @classmethod
    def load_state_dict(cls, path: str) -> "UMAPMixture":
        """Load a saved model from disk.

        Args:
            path: Path to the saved checkpoint.

        Returns:
            Restored UMAPMixture model ready for inference.
        """
        state_dict = torch.load(path)

        model = cls(
            k_neighbors=state_dict['k_neighbors'],
            out_dim=state_dict['out_dim'],
            min_dist=state_dict['min_dist'],
            num_encoders=state_dict['num_encoders'],
        )

        model.a = state_dict['a']
        model.b = state_dict['b']

        for encoder, encoder_state in zip(model.encoders, state_dict['encoders']):
            encoder.sigmas = encoder_state['sigmas']
            encoder.rhos = encoder_state['rhos']

        model.data = state_dict['data']
        model.graphs = state_dict['graphs']
        model.embeds = state_dict['embeds']

        return model