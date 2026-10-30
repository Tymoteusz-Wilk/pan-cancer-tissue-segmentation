import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

# Two-stage training:
# Stage 1: Train each expert head independently to learn their individual strengths
# Stage 2: Train cross-expert attention + embedding-conditioned routing (FMoE)

HIDDEN_DIM = 128  # Shared hidden dimension across experts


# ---------------------------
# Stage 1: Per-expert linear probe
# ---------------------------

class LinearProbe(nn.Module):
    """Per-expert head: maps a frozen backbone embedding to class logits.
    
    Architecture: embedding → Linear(hidden) → GELU → Dropout → Linear(classes)
    
    The encoder/classifier split enables FMoE to reuse learned
    representations while adding cross-expert communication in Stage 2.
    """
    def __init__(self, num_classes: int, embedding_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden representation (before classifier)."""
        return self.encoder(x)


# ---------------------------
# Stage 2: FMoE — Foundation Model Mixture of Experts
# ---------------------------

class CrossExpertAttention(nn.Module):
    """Transformer-style self-attention over expert hidden representations.
    
    Enables experts to communicate: each expert's hidden representation
    attends to all other experts' representations for the same sample.
    This captures complementary signals across foundation models that were
    pre-trained on different data / with different objectives.
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, num_experts, hidden_dim)
        Returns:
            x: (batch, num_experts, hidden_dim) — refined representations
            attn_weights: (batch, num_experts, num_experts) — attention map
        """
        residual = x
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(residual + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class EmbeddingRouter(nn.Module):
    """Input-dependent expert routing from raw foundation model embeddings.
    
    Each expert's raw embedding is projected to a scalar routing score.
    Scores are normalized via temperature-scaled softmax to produce
    per-sample mixture weights.
    """
    def __init__(self, expert_dims: List[int], num_experts: int, router_dim: int = 64):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, router_dim),
                nn.GELU(),
                nn.Linear(router_dim, 1),
            )
            for dim in expert_dims
        ])
        self.temperature = nn.Parameter(torch.tensor(float(np.log(num_experts))))

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_list: list of (batch, emb_dim_i) raw embeddings per expert
        Returns:
            weights: (batch, num_experts) soft routing weights
        """
        scores = torch.cat(
            [proj(x) for proj, x in zip(self.projections, x_list)], dim=-1
        )
        return F.softmax(scores / F.softplus(self.temperature), dim=-1)


class FMoE(nn.Module):
    """Foundation Model Mixture of Experts with Cross-Expert Attention.
    
    Three components over standard ensembling:
    
    1. Cross-Expert Attention — experts communicate in a shared hidden space
       before making predictions, enabling complementary signal integration
       across heterogeneous foundation models.
       
    2. Embedding-Conditioned Router — a lightweight network produces
       per-sample expert mixture weights from raw embeddings, creating
       true input-dependent expert selection.
       
    3. Diversity Loss — encourages expert specialization (low per-sample
       routing entropy) while preventing expert collapse (uniform average
       load across the dataset). Inspired by Switch Transformer.
    
    Architecture:
        Raw embeddings → Per-expert encoders (warm-started from Stage 1)
        → Cross-expert attention (experts communicate)
        → Per-expert classifiers (warm-started from Stage 1)
        → Embedding router produces per-sample mixture weights
        → Weighted aggregation of expert logits → final prediction
    """

    def __init__(
        self,
        experts: List[LinearProbe],
        expert_embedding_dims: List[int],
        num_heads: int = 4,
        num_attn_layers: int = 2,
        router_dim: int = 64,
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.num_classes = experts[0].num_classes
        hidden_dim = experts[0].hidden_dim
        self.hidden_dim = hidden_dim

        # Warm-start from pre-trained Stage 1 experts
        self.encoders = nn.ModuleList([e.encoder for e in experts])
        self.classifiers = nn.ModuleList([e.classifier for e in experts])

        # Cross-expert attention (NEW — learned from scratch in Stage 2)
        self.cross_attention = nn.ModuleList([
            CrossExpertAttention(hidden_dim, num_heads)
            for _ in range(num_attn_layers)
        ])

        # Embedding-conditioned router (NEW — learned from scratch in Stage 2)
        self.router = EmbeddingRouter(expert_embedding_dims, self.num_experts, router_dim)

        # Per-expert temperature scaling
        self.temperatures = nn.Parameter(torch.ones(self.num_experts))

    def forward(
        self,
        x_list: List[torch.Tensor],
        return_routing: bool = False,
    ):
        """
        Args:
            x_list: list of (batch, emb_dim_i) raw embeddings per expert
            return_routing: also return routing weights & attention maps
        Returns:
            logits: (batch, num_classes)
            [routing_weights]: (batch, num_experts)
            [attn_weights]: list of (batch, num_experts, num_experts)
        """
        # 1. Per-expert encoding (warm-started, fine-tuned)
        hiddens = [enc(x) for enc, x in zip(self.encoders, x_list)]
        stacked = torch.stack(hiddens, dim=1)  # (B, E, H)

        # 2. Cross-expert attention (experts communicate)
        attn_weights_all = []
        for layer in self.cross_attention:
            stacked, attn_w = layer(stacked)
            attn_weights_all.append(attn_w)

        # 3. Per-expert classification from attended representations
        expert_logits = []
        for i, clf in enumerate(self.classifiers):
            logits = clf(stacked[:, i]) / F.softplus(self.temperatures[i])

            expert_logits.append(logits)
        stacked_logits = torch.stack(expert_logits, dim=1)  # (B, E, C)

        # 4. Embedding-conditioned routing
        routing_weights = self.router(x_list)  # (B, E)

        # 5. Weighted aggregation
        final = (stacked_logits * routing_weights.unsqueeze(-1)).sum(dim=1)

        if return_routing:
            return final, routing_weights, attn_weights_all
        return final

    def diversity_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Expert specialization + load balancing regularization.
        
        Minimizing this encourages:
          - Low per-sample routing entropy → each sample prefers specific experts
          - Uniform average expert load → no dead or over-used experts
        """
        eps = 1e-8
        # Specialization: minimize per-sample entropy (peaky routing)
        entropy = -(routing_weights * (routing_weights + eps).log()).sum(dim=-1).mean()

        # Load balance: average routing per expert should be uniform
        avg_load = routing_weights.mean(dim=0)
        target = torch.ones_like(avg_load) / self.num_experts
        balance = F.mse_loss(avg_load, target)

        return entropy + 5.0 * balance


class LogitCrossAttention(nn.Module):
    """Simple self-attention over per-expert logits — residual + norm, no FFN.
    
    num_heads must divide num_classes; the largest valid divisor ≤ requested
    num_heads is chosen automatically.
    """
    def __init__(self, num_classes: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        # num_heads must divide num_classes
        num_heads = 1
        self.attn = nn.MultiheadAttention(
            num_classes, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(num_classes)
        # Zero-init out_proj so the module starts as identity (LN(x + 0) = LN(x)).
        # This prevents the randomly-initialised attention from corrupting well-trained
        # logits at the start of Stage 2.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, logits: torch.Tensor):
        """
        Args:
            logits: (batch, num_experts, num_classes)
        Returns:
            logits: (batch, num_experts, num_classes) — enriched
            attn_weights: (batch, num_experts, num_experts)
        """
        attn_out, attn_weights = self.attn(logits, logits, logits)
        return logits + self.norm(attn_out), attn_weights


class EmbeddingCrossAttention(nn.Module):
    """Simple self-attention over raw expert embeddings in a shared space.
    
    proj_in → attention → residual + norm → proj_out + residual. No FFN.
    """
    def __init__(
        self,
        expert_embedding_dims: List[int],
        shared_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_in = nn.ModuleList([
            nn.Linear(dim, shared_dim) for dim in expert_embedding_dims
        ])
        self.attn = nn.MultiheadAttention(
            shared_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(shared_dim)
        self.proj_out = nn.ModuleList([
            nn.Linear(shared_dim, dim) for dim in expert_embedding_dims
        ])
        # Zero-init proj_out so enrichment starts at 0 (residual = x_orig unchanged).
        for proj in self.proj_out:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x_list: List[torch.Tensor]):
        """
        Args:
            x_list: list of (batch, emb_dim_i)
        Returns:
            enriched_list: list of (batch, emb_dim_i)
            attn_weights: (batch, num_experts, num_experts)
        """
        shared = torch.stack(
            [proj(x) for proj, x in zip(self.proj_in, x_list)], dim=1
        )  # (B, E, shared_dim)
        attn_out, attn_weights = self.attn(shared, shared, shared)
        shared = self.norm(shared + attn_out)
        enriched_list = [
            proj_out(shared[:, i]) + x_orig
            for i, (proj_out, x_orig) in enumerate(zip(self.proj_out, x_list))
        ]
        return enriched_list, attn_weights


class HiddenCrossAttention(nn.Module):
    """Simple self-attention over per-expert hidden representations.
    
    All experts share the same hidden_dim so no projection is needed —
    just attention + residual + norm directly in hidden space.
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # Zero-init out_proj so the module starts as identity.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, hidden_list: List[torch.Tensor]):
        """
        Args:
            hidden_list: list of (batch, hidden_dim) per-expert hidden representations
        Returns:
            enriched_list: list of (batch, hidden_dim) enriched hidden representations
            attn_weights: (batch, num_experts, num_experts)
        """
        stacked = torch.stack(hidden_list, dim=1)  # (B, E, H)
        attn_out, attn_weights = self.attn(stacked, stacked, stacked)
        stacked = stacked + self.norm(attn_out)  # norm the delta; zero out_proj → identity
        return [stacked[:, i] for i in range(len(hidden_list))], attn_weights


class CrossAttentionEnsemble(nn.Module):
    """Ensemble with optional cross-attention at a configurable layer.
    
    Architecture:
        Raw embeddings
        → [EmbeddingCrossAttention]   (if cross_attention_layer="embedding")
        → Per-expert encoders
        → [HiddenCrossAttention]      (if cross_attention_layer="hidden")
        → Per-expert classifiers
        → [LogitCrossAttention]       (if cross_attention_layer="logit")
        → Mean of logits

    cross_attention_layer options:
        "none"      — fully independent experts (ablation baseline)
        "embedding" — experts share information at raw embedding level
        "hidden"    — experts share information after encoding (default)
        "logit"     — experts share information at prediction level
    """
    def __init__(
        self,
        num_classes: int,
        expert_embedding_dims: List[int],
        hidden_dim: int = HIDDEN_DIM
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = len(expert_embedding_dims)
        self.expert_embedding_dims = expert_embedding_dims
        self.hidden_dim = hidden_dim

        # Per-expert encoders: embedding → hidden
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
            for dim in expert_embedding_dims
        ])

        # Per-expert classifiers: hidden → logits
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in expert_embedding_dims
        ])

        # Learnable per-expert temperature scaling
        self.temperatures = nn.Parameter(torch.ones(self.num_experts))

        # Stage 2: cross-expert attention at the raw embedding level.
        # Zero-init proj_out → identity at init, so Stage 1 quality is fully preserved.
        # NOT included in trainable_params() — activated only in Stage 2 via its own param group.
        self.cross_attention = EmbeddingCrossAttention(expert_embedding_dims)

    def _compute_logits(self, x_list: List[torch.Tensor]):
        """Full forward pass; returns expert_logits_list."""
        # --- cross-expert communication at embedding level (identity when zero-init / Stage 1) ---
        x_list, _ = self.cross_attention(x_list)

        # --- encode to hidden ---
        hidden_list = [enc(x) for enc, x in zip(self.encoders, x_list)]

        # --- classify ---
        logits = [
            clf(h) / F.softplus(self.temperatures[i])
            for i, (clf, h) in enumerate(zip(self.classifiers, hidden_list))
        ]

        return logits

    def forward_single_expert(self, x_list: List[torch.Tensor], expert_idx: int, use_cross_attention: bool = False, use_temperature: bool = True):
        """Alternating-training forward for a single expert.
        
        Args:
            x_list: raw embeddings for all experts (needed when use_cross_attention=True).
            expert_idx: which expert's logits to return.
            use_cross_attention: if True, all embeddings pass through EmbeddingCrossAttention
                first; only expert_idx's enriched embedding is then encoded + classified.
                Other experts' parameters are frozen by the caller so they receive no
                gradient updates, but CA parameters always do.
                If False (default), CA is bypassed entirely — pure Stage 1 behaviour.
        """
        if use_cross_attention:
            x_list, _ = self.cross_attention(x_list)
        x = x_list[expert_idx]
        h = self.encoders[expert_idx](x)
        return self.classifiers[expert_idx](h) / F.softplus(self.temperatures[expert_idx]) if use_temperature else self.classifiers[expert_idx](h)

    def forward_all_experts(self, x_list: List[torch.Tensor]):
        """Forward for all experts; returns list of per-expert logit tensors."""
        logits = self._compute_logits(x_list)
        return logits

    def forward(self, x_list: List[torch.Tensor], return_attention: bool = False):
        """Mean of per-expert logits."""
        logits = self._compute_logits(x_list)
        out = torch.stack(logits, dim=0).mean(dim=0)  # (B, C)
        return out

    def trainable_params(self, include_cross_attention: bool = False, include_temperature: bool = True):
        """All trainable parameters.
        
        Args:
            include_cross_attention: if True, also include cross_attention parameters.
                                     Set True to train CA jointly in Stage 1 (no Stage 2 needed).
        """
        params = (list(self.encoders.parameters()) +
                  list(self.classifiers.parameters()))
        if include_cross_attention:
            params += list(self.cross_attention.parameters())
        if include_temperature:
            params += [self.temperatures]
        return params


class LogitMLPEnsemble(nn.Module):
    """
    Ablation: Simple MLP that takes in the logits from all experts and returns a final prediction.
    Use this instead of the cross-expert attention.
    """
    def __init__(
        self,
        num_classes: int,
        expert_embedding_dims: List[int],
        hidden_dim: int = HIDDEN_DIM
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = len(expert_embedding_dims)
        self.expert_embedding_dims = expert_embedding_dims
        self.hidden_dim = hidden_dim

        # Per-expert encoders: embedding → hidden (independent)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
            for dim in expert_embedding_dims
        ])

        # Per-expert classifiers: hidden → logits (independent)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in expert_embedding_dims
        ])

        # Learnable per-expert temperature scaling
        self.temperatures = nn.Parameter(torch.ones(self.num_experts))

        # Ablation: MLP on top of concatenated logits
        self.mlp = nn.Sequential(
            nn.Linear(self.num_experts * num_classes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def _compute_logits(self, x_list: List[torch.Tensor]):
        """Full forward pass to get expert logits; returns expert_logits_list."""
        # --- encode to hidden ---
        hidden_list = [enc(x) for enc, x in zip(self.encoders, x_list)]

        # --- classify ---
        logits = [
            clf(h) / F.softplus(self.temperatures[i])
            for i, (clf, h) in enumerate(zip(self.classifiers, hidden_list))
        ]

        return logits

    def forward_single_expert(self, x_list: List[torch.Tensor], expert_idx: int, use_cross_attention: bool = False, use_temperature: bool = True):
        # Simply ignore use_cross_attention since we don't have it.
        x = x_list[expert_idx]
        h = self.encoders[expert_idx](x)
        return self.classifiers[expert_idx](h) / F.softplus(self.temperatures[expert_idx]) if use_temperature else self.classifiers[expert_idx](h)

    def forward_all_experts(self, x_list: List[torch.Tensor]):
        return self._compute_logits(x_list)

    def forward(self, x_list: List[torch.Tensor], return_attention: bool = False):
        """Concatenate logits + MLP."""
        logits_list = self._compute_logits(x_list)
        cat_logits = torch.cat(logits_list, dim=1) # (B, E*C)
        return self.mlp(cat_logits)

    def trainable_params(self, include_cross_attention: bool = False, include_temperature: bool = True):
        # include_cross_attention controls whether to include the mixing MLP
        params = (list(self.encoders.parameters()) +
                  list(self.classifiers.parameters()))
        if include_cross_attention:
            params += list(self.mlp.parameters())
        if include_temperature:
            params += [self.temperatures]
        return params


class GatedAverageEnsemble(nn.Module):
    """Adaptive weighted averaging over expert logits using hidden-feature gates."""
    def __init__(
        self,
        num_classes: int,
        expert_embedding_dims: List[int],
        hidden_dim: int = HIDDEN_DIM
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = len(expert_embedding_dims)
        self.expert_embedding_dims = expert_embedding_dims
        self.hidden_dim = hidden_dim

        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
            for dim in expert_embedding_dims
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in expert_embedding_dims
        ])
        self.temperatures = nn.Parameter(torch.ones(self.num_experts))

        gate_input_dim = self.num_experts * hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_experts),
        )

    def _compute_logits(self, x_list: List[torch.Tensor]):
        hidden_list = [enc(x) for enc, x in zip(self.encoders, x_list)]
        return [
            clf(h) / F.softplus(self.temperatures[i])
            for i, (clf, h) in enumerate(zip(self.classifiers, hidden_list))
        ]

    def _compute_gate_weights(self, hidden_list: List[torch.Tensor]):
        gate_input = torch.cat(hidden_list, dim=1) # (B, E*H)
        return F.softmax(self.gate(gate_input), dim=1) # (B, E)

    def forward_single_expert(self, x_list: List[torch.Tensor], expert_idx: int, use_cross_attention: bool = False, use_temperature: bool = True):
        x = x_list[expert_idx]
        h = self.encoders[expert_idx](x)
        logits = self.classifiers[expert_idx](h)
        if use_temperature:
            logits = logits / F.softplus(self.temperatures[expert_idx])
        return logits

    def forward_all_experts(self, x_list: List[torch.Tensor]):
        return self._compute_logits(x_list)

    def forward(self, x_list: List[torch.Tensor], return_attention: bool = False):
        hidden_list = [enc(x) for enc, x in zip(self.encoders, x_list)]
        logits_list = [
            clf(h) / F.softplus(self.temperatures[i])
            for i, (clf, h) in enumerate(zip(self.classifiers, hidden_list))
        ]
        weights = self._compute_gate_weights(hidden_list)
        stacked_logits = torch.stack(logits_list, dim=1) # (B, E, C)
        return (stacked_logits * weights.unsqueeze(-1)).sum(dim=1)

    def trainable_params(self, include_cross_attention: bool = False, include_temperature: bool = True):
        params = (list(self.encoders.parameters()) +
                  list(self.classifiers.parameters()))
        if include_cross_attention:
            params += list(self.gate.parameters())
        if include_temperature:
            params += [self.temperatures]
        return params


class ConfidenceWeightedEnsemble(nn.Module):
    """Weighted logit averaging where expert weights come from prediction confidence."""
    def __init__(
        self,
        num_classes: int,
        expert_embedding_dims: List[int],
        hidden_dim: int = HIDDEN_DIM
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = len(expert_embedding_dims)
        self.expert_embedding_dims = expert_embedding_dims
        self.hidden_dim = hidden_dim

        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
            for dim in expert_embedding_dims
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in expert_embedding_dims
        ])
        self.temperatures = nn.Parameter(torch.ones(self.num_experts))

        # Scales confidence-derived routing sharpness while keeping the rule heuristic.
        self.confidence_temperature = nn.Parameter(torch.tensor(1.0))

    def _compute_logits(self, x_list: List[torch.Tensor]):
        hidden_list = [enc(x) for enc, x in zip(self.encoders, x_list)]
        return [
            clf(h) / F.softplus(self.temperatures[i])
            for i, (clf, h) in enumerate(zip(self.classifiers, hidden_list))
        ]

    def _compute_confidence_weights(self, logits_list: List[torch.Tensor]):
        confidences = []
        for logits in logits_list:
            probs = torch.softmax(logits, dim=1)
            confidences.append(probs.max(dim=1).values) # (B,)
        confidence_scores = torch.stack(confidences, dim=1)  # (B, E)
        return F.softmax(confidence_scores / F.softplus(self.confidence_temperature), dim=1)

    def forward_single_expert(self, x_list: List[torch.Tensor], expert_idx: int, use_cross_attention: bool = False, use_temperature: bool = True):
        x = x_list[expert_idx]
        h = self.encoders[expert_idx](x)
        logits = self.classifiers[expert_idx](h)
        if use_temperature:
            logits = logits / F.softplus(self.temperatures[expert_idx])
        return logits

    def forward_all_experts(self, x_list: List[torch.Tensor]):
        return self._compute_logits(x_list)

    def forward(self, x_list: List[torch.Tensor], return_attention: bool = False):
        logits_list = self._compute_logits(x_list)
        weights = self._compute_confidence_weights(logits_list) # (B, E)
        stacked_logits = torch.stack(logits_list, dim=1) # (B, E, C)
        return (stacked_logits * weights.unsqueeze(-1)).sum(dim=1) # (B, C)

    def trainable_params(self, include_cross_attention: bool = False, include_temperature: bool = True):
        params = (list(self.encoders.parameters()) +
                  list(self.classifiers.parameters()))
        if include_cross_attention:
            params += [self.confidence_temperature]
        if include_temperature:
            params += [self.temperatures]
        return params


class OrganCrossAttentionEnsemble(CrossAttentionEnsemble):
    """
    Ablation: CrossAttentionEnsemble that also takes an organ ID, embeds it,
    and concatenates the organ embedding to the input of both the cross-expert
    attention block and the expert MLPs (Encoders).
    
    Inherits from CrossAttentionEnsemble to reuse its forward logic.
    """
    def __init__(
        self,
        num_classes: int,
        expert_embedding_dims: List[int],
        num_organs: int,
        organ_embedding_dim: int = 64,
        hidden_dim: int = HIDDEN_DIM
    ):
        # We augment the expected input dimensions for the base class
        # But we must init `self.organ_embedding` properly.
        # Since we use `super().__init__` which builds submodules based on `expert_embedding_dims`,
        # we pass the augmented dims to `super()`.
        
        aug_dims = [d + organ_embedding_dim for d in expert_embedding_dims]
        
        super().__init__(
            num_classes=num_classes,
            expert_embedding_dims=aug_dims,
            hidden_dim=hidden_dim
        )
        
        self.organ_embedding_dim = organ_embedding_dim
        self.organ_embedding = nn.Embedding(num_organs, organ_embedding_dim)

    def forward_single_expert(self, x_list: List[torch.Tensor], expert_idx: int, organ_ids: torch.Tensor, use_cross_attention: bool = False, use_temperature: bool = True):
        organ_emb = self.organ_embedding(organ_ids) # (B, org_dim)
        x_list_aug = [torch.cat([x, organ_emb], dim=1) for x in x_list]
        return super().forward_single_expert(x_list_aug, expert_idx, use_cross_attention, use_temperature)

    def forward_all_experts(self, x_list: List[torch.Tensor], organ_ids: torch.Tensor):
        organ_emb = self.organ_embedding(organ_ids)
        x_list_aug = [torch.cat([x, organ_emb], dim=1) for x in x_list]
        return super().forward_all_experts(x_list_aug)

    def forward(self, x_list: List[torch.Tensor], organ_ids: torch.Tensor, return_attention: bool = False):
        organ_emb = self.organ_embedding(organ_ids)
        x_list_aug = [torch.cat([x, organ_emb], dim=1) for x in x_list]
        return super().forward(x_list_aug, return_attention)

    def trainable_params(self, include_cross_attention: bool = False, include_temperature: bool = True):
        params = super().trainable_params(include_cross_attention, include_temperature)
        params += list(self.organ_embedding.parameters())
        return params


class GatedEnsembleModel(nn.Module):
    """Expert ensemble with residual connections (legacy model).
    
    Novel contribution: Residual Expert Averaging
    - Simple meta-classifier learns to combine expert predictions
    - Residual connection preserves individual expert knowledge
    - Learnable weight α balances meta-classifier vs expert averaging
    
    Architecture:
        Raw embeddings → Per-expert encoders (warm-started from Stage 1)
        → Per-expert classifiers (warm-started from Stage 1)
        → Temperature scaling → Meta-classifier → Final prediction
        → Residual connection from averaged expert predictions
    
    Final output: meta_out + α * mean(expert_predictions)
    """
    def __init__(
        self, 
        experts: List[LinearProbe], 
        expert_embedding_dims: List[int],
        **kwargs  # Ignore unused attention params for backward compatibility
    ):
        super().__init__()
        self.num_classes = experts[0].num_classes
        self.num_experts = len(experts)
        self.expert_embedding_dims = expert_embedding_dims
        self.hidden_dim = experts[0].hidden_dim
        
        # Warm-start encoders and classifiers from pre-trained Stage 1 experts
        self.encoders = nn.ModuleList([e.encoder for e in experts])
        self.classifiers = nn.ModuleList([e.classifier for e in experts])
        
        # Learnable temperature scaling for each expert
        self.temperatures = nn.Parameter(torch.ones(self.num_experts))
        
        # Meta-classifier operates on concatenated expert logits
        meta_input_dim = self.num_experts * self.num_classes
        self.meta_classifier = nn.Sequential(
            nn.Linear(meta_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.num_classes)
        )
        
        # Residual connection weight (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize to 0.5 (equal weight)
        
    def forward(self, x_list: List[torch.Tensor]):
        """
        Args:
            x_list: list of (batch, emb_dim_i) raw embeddings per expert
        Returns:
            logits: (batch, num_classes)
        """
        # 1. Per-expert encoding and classification (warm-started, fine-tuned)
        expert_logits = []
        for i, (enc, clf, x) in enumerate(zip(self.encoders, self.classifiers, x_list)):
            hidden = enc(x)
            logits = clf(hidden) / F.softplus(self.temperatures[i])
            expert_logits.append(logits)
        
        # 2. Stack expert predictions for averaging
        expert_logits_stacked = torch.stack(expert_logits, dim=0)  # (E, B, C)
        expert_avg = expert_logits_stacked.mean(dim=0)  # (B, C)
        
        # 3. Concatenate for meta-classifier: (B, E*C)
        all_logits = torch.cat(expert_logits, dim=1)
        
        # 4. Meta-classifier prediction
        meta_out = self.meta_classifier(all_logits)
        
        # 5. Residual connection: final = meta_out + α * expert_avg
        final_logits = meta_out + self.alpha * expert_avg
        
        return final_logits
