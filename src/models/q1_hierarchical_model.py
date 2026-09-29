from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, lambda_)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class ResidualShrinkageBlock(nn.Module):
    """Soft-threshold denoising inspired by deep residual shrinkage networks."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        inner = max(dim // reduction, 8)
        self.threshold = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.detach().abs().mean(dim=1, keepdim=True)
        threshold = self.threshold(scale) * scale
        denoised = torch.sign(x) * F.relu(x.abs() - threshold)
        return x + denoised


class CPGGenePathwayAttentionNet(nn.Module):
    """Hierarchical CpG -> gene -> pathway attention model.

    The model expects fixed-size index maps created during preprocessing:
    - gene_cpg_index: [n_genes, max_cpg_per_gene], values are feature indices or -1 padding.
    - pathway_gene_index: [n_pathways, max_gene_per_pathway], values are gene indices or -1 padding.

    It returns class logits plus optional domain logits for adversarial platform/batch alignment.
    """

    def __init__(
        self,
        gene_cpg_index: torch.Tensor,
        pathway_gene_index: torch.Tensor,
        num_classes: int,
        num_domains: int = 0,
        num_features: int | None = None,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if gene_cpg_index.ndim != 2 or pathway_gene_index.ndim != 2:
            raise ValueError("Index maps must be 2D tensors")
        self.register_buffer("gene_cpg_index", gene_cpg_index.long())
        self.register_buffer("pathway_gene_index", pathway_gene_index.long())
        self.num_classes = num_classes
        self.num_domains = num_domains
        n_features = int(gene_cpg_index.max().item()) + 1
        if num_features is not None:
            n_features = max(n_features, int(num_features))
        n_genes = int(gene_cpg_index.shape[0])
        n_pathways = int(pathway_gene_index.shape[0])

        self.cpg_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.cpg_identity = nn.Embedding(n_features, hidden_dim)
        self.gene_identity = nn.Embedding(n_genes, hidden_dim)
        self.pathway_identity = nn.Embedding(n_pathways, hidden_dim)
        self.flat_cpg_pool = MaskedAttentionPool(hidden_dim)
        self.flat_project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.gene_pool = MaskedAttentionPool(hidden_dim)
        self.gene_project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.gene_denoise = ResidualShrinkageBlock(hidden_dim)
        self.pathway_pool = MaskedAttentionPool(hidden_dim)
        self.pathway_project = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.pathway_denoise = ResidualShrinkageBlock(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_classifier = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_domains),
            )
            if num_domains > 1
            else None
        )

    def _gather_with_padding(self, values: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = index >= 0
        safe_index = index.clamp_min(0)
        gathered = values[:, safe_index]
        gathered = gathered * mask.unsqueeze(0).unsqueeze(-1).to(gathered.dtype)
        return gathered, mask

    def forward(self, beta_values: torch.Tensor, grl_lambda: float = 1.0) -> dict[str, torch.Tensor]:
        cpg_embed = self.cpg_encoder(beta_values.unsqueeze(-1))
        cpg_ids = torch.arange(beta_values.shape[1], device=beta_values.device)
        cpg_embed = cpg_embed + self.cpg_identity(cpg_ids).unsqueeze(0)
        flat_embed, flat_cpg_attention = self.flat_cpg_pool(cpg_embed)
        flat_embed = self.flat_project(flat_embed)

        gene_cpg_embed, gene_cpg_mask = self._gather_with_padding(cpg_embed, self.gene_cpg_index)
        batch, n_genes, max_cpg, hidden = gene_cpg_embed.shape
        gene_flat = gene_cpg_embed.reshape(batch * n_genes, max_cpg, hidden)
        gene_mask_flat = gene_cpg_mask.reshape(1, n_genes, max_cpg).expand(batch, -1, -1).reshape(batch * n_genes, max_cpg)
        gene_embed, cpg_attention = self.gene_pool(gene_flat, gene_mask_flat)
        gene_embed = self.gene_project(gene_embed.reshape(batch, n_genes, hidden))
        gene_ids = torch.arange(n_genes, device=beta_values.device)
        gene_embed = gene_embed + self.gene_identity(gene_ids).unsqueeze(0)
        gene_embed = self.gene_denoise(gene_embed)

        pathway_gene_embed, pathway_gene_mask = self._gather_with_padding(gene_embed, self.pathway_gene_index)
        _, n_pathways, max_genes, _ = pathway_gene_embed.shape
        pathway_flat = pathway_gene_embed.reshape(batch * n_pathways, max_genes, hidden)
        pathway_mask_flat = pathway_gene_mask.reshape(1, n_pathways, max_genes).expand(batch, -1, -1).reshape(batch * n_pathways, max_genes)
        pathway_embed, gene_attention = self.pathway_pool(pathway_flat, pathway_mask_flat)
        pathway_embed = self.pathway_project(pathway_embed.reshape(batch, n_pathways, hidden))
        pathway_ids = torch.arange(n_pathways, device=beta_values.device)
        pathway_embed = pathway_embed + self.pathway_identity(pathway_ids).unsqueeze(0)
        pathway_embed = self.pathway_denoise(pathway_embed)

        sample_embed, pathway_attention = self.pathway_pool(pathway_embed)
        fused_embed = self.fusion(torch.cat([sample_embed, flat_embed], dim=-1))
        class_logits = self.classifier(fused_embed)
        outputs = {
            "class_logits": class_logits,
            "sample_embedding": fused_embed,
            "hierarchical_embedding": sample_embed,
            "flat_cpg_embedding": flat_embed,
            "flat_cpg_attention": flat_cpg_attention,
            "cpg_attention": cpg_attention.reshape(batch, n_genes, max_cpg),
            "gene_attention": gene_attention.reshape(batch, n_pathways, max_genes),
            "pathway_attention": pathway_attention,
        }
        if self.domain_classifier is not None:
            outputs["domain_logits"] = self.domain_classifier(gradient_reverse(fused_embed, grl_lambda))
        return outputs


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_tensor(0.0)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (source.shape[0] - 1)
    target_cov = target_centered.T @ target_centered / (target.shape[0] - 1)
    return torch.mean((source_cov - target_cov) ** 2)


def classification_domain_loss(
    outputs: dict[str, torch.Tensor],
    class_targets: torch.Tensor,
    domain_targets: torch.Tensor | None = None,
    domain_weight: float = 0.1,
) -> torch.Tensor:
    loss = F.cross_entropy(outputs["class_logits"], class_targets)
    if domain_targets is not None and "domain_logits" in outputs:
        loss = loss + domain_weight * F.cross_entropy(outputs["domain_logits"], domain_targets)
    return loss
