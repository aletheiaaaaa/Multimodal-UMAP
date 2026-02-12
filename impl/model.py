import torch
import os
import numpy as np
from pynndescent import NNDescent
from torch import linalg as LA
from torch import sparse as sp
from torch import autograd
from torch import optim
from torch.optim.lr_scheduler import LinearLR
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

        return sigmas.detach()

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

        inputs_np = inputs.cpu().numpy()

        if query is None:
            index = NNDescent(inputs_np, n_neighbors=self.k_neighbors, metric="euclidean")
            indices_np, dists_np = index.neighbor_graph
        else:
            query_np = query.cpu().numpy()
            index = NNDescent(inputs_np, n_neighbors=self.k_neighbors, metric="euclidean")
            indices_np, dists_np = index.query(query_np, k=self.k_neighbors)

        cols = torch.from_numpy(indices_np.flatten()).long().to(device)
        rows = torch.arange(Q).repeat_interleave(self.k_neighbors).to(device)
        dists = torch.from_numpy(dists_np).float().to(device)
        dists = dists.view(Q, self.k_neighbors)
        if mode != "invert":
            min_dists = dists.min(dim=1).values.unsqueeze(1)
            sigmas = self.get_sigmas(dists, min_dists)
            weights = torch.exp(- (dists - min_dists) / sigmas.unsqueeze(1)).flatten()
            if mode == "fit":
                self.sigmas = sigmas
                self.rhos = min_dists[:, 0]
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

    def _mse_loss(self, embeds_0: torch.Tensor, embeds_1: torch.Tensor) -> torch.Tensor:
        n = min(embeds_0.size(0), embeds_1.size(0))
        return (embeds_0[:n] - embeds_1[:n]).pow(2).sum(dim=1).mean()

    def _infonce_loss(self, embeds_0: torch.Tensor, embeds_1: torch.Tensor, n_neg: int = 128, temperature: float = 0.1) -> torch.Tensor:
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

        optimizer = optim.Adam(embeds, lr=lr)
        # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)

        pbar = tqdm(range(epochs), desc=desc)
        for epoch in pbar:
            embed_losses = []
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

                count = embed.size(0)
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
                    rep_count = ref_embed.size(0) if ref_embed is not None else count
                    l_idx_rep = torch.randint(0, rep_count, (num_pairs, num_rep)).flatten().to(device)

                    if mode == "invert":
                        loss_rep = self._inv_rep_loss(embed, i_idx_rep, l_idx_rep, ref_embed, self.encoders[i].sigmas, self.encoders[i].rhos)
                    else:
                        loss_rep = self._umap_rep_loss(embed, i_idx_rep, l_idx_rep, self.a, self.b, ref_embed)

                    batch_losses.append(loss_attr + loss_rep)

                embed_loss = torch.stack(batch_losses).mean()
                embed_losses.append(embed_loss)

            loss = sum(embed_losses)

            # Compute InfoNCE losses between modalities for cross-modal alignment
            if mode == "fit":
                num_embeds = len(embeds)
                align_losses = [torch.tensor(0.0, requires_grad=True).to(device) for _ in range(num_embeds)]

                for i in range(num_embeds):
                    for j in range(i + 1, num_embeds):
                        align_loss = self._mse_loss(embeds[i], embeds[j])

                        align_losses[i] = align_losses[i] + alpha * align_loss
                        align_losses[j] = align_losses[j] + alpha * align_loss

                loss += sum(align_losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if epoch % 10 == 0 or epoch == epochs - 1:
                umap_total = sum(embed_losses).item()
                align_total = (loss.item() - umap_total) if mode == "fit" else 0.0
                pbar.set_description(desc=f"{desc} (umap: {umap_total:.4f}, align: {align_total:.4f})")

        if mode == "transform" and len(embeds) >= 2:
            e0 = F.normalize(embeds[0].detach(), dim=1)
            e1 = F.normalize(embeds[1].detach(), dim=1)
            final_cossim = (e0 * e1).sum(dim=1).mean().item()
            print(f"[diag] final cossim (after optimization): {final_cossim:.4f}")

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

        if len(self.embeds) >= 2:
            e0 = F.normalize(self.embeds[0].detach(), dim=1)
            e1 = F.normalize(self.embeds[1].detach(), dim=1)
            print(f"[diag] fit training embed cossim: {(e0 * e1).sum(dim=1).mean().item():.4f}")

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

        # Diagnostics
        if len(embeds) >= 2:
            e0 = F.normalize(embeds[0], dim=1)
            e1 = F.normalize(embeds[1], dim=1)
            print(f"[diag] init embed cossim: {(e0 * e1).sum(dim=1).mean().item():.4f}")

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
                graph, embed = encoder.init(self.embeds[i], mode="invert", query=inputs[idx], ref_data=self.graphs[i], a=self.a, b=self.b)
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

        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

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

        model = cls.__new__(cls)
        model.k_neighbors = state_dict['k_neighbors']
        model.out_dim = state_dict['out_dim']
        model.min_dist = state_dict['min_dist']
        model.num_encoders = state_dict['num_encoders']
        model.a = state_dict['a']
        model.b = state_dict['b']

        model.encoders = [UMAPEncoder(model.k_neighbors, model.out_dim, id=i) for i in range(model.num_encoders)]
        for encoder, encoder_state in zip(model.encoders, state_dict['encoders']):
            encoder.sigmas = encoder_state['sigmas']
            encoder.rhos = encoder_state['rhos']

        model.data = state_dict['data']
        model.graphs = state_dict['graphs']
        model.embeds = state_dict['embeds']

        return model