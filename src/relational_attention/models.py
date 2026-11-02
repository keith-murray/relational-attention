import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, context_size, linear=False):
        super().__init__()
        # Flag for linear attention
        self.linear = linear

        # Linear projections for queries, keys, and values
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        returns: (batch_size, seq_len, head_dim)
        """
        # Compute Q, K, V
        Q = self.W_q(x)   # (B, L, D_h)
        K = self.W_k(x)   # (B, L, D_h)
        V = self.W_v(x)   # (B, L, D_h)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # (B, L, L)
        if not self.linear:
            attn_weights = F.softmax(attn_weights, dim=-1)  # attention probabilities
        out = torch.bmm(attn_weights, V)  # (B, L, D_h)

        return out, attn_weights


class RelationalAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, context_size, linear=False):
        super().__init__()
        # Flag for linear attention
        self.linear = linear

        # Initialize weights
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)

        # Learnable Value matrix (not derived from input)
        self.V = nn.Parameter(torch.randn(context_size, head_dim) * 0.1)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        returns: (batch, seq_len, head_dim)
        """
        Q = self.W_q(x)                            # (B, L, D)
        K = self.W_k(x)                            # (B, L, D)

        # Compute attention scores over the *learned* Value slots
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # (B, L, L)
        if not self.linear:
            attn_weights = F.softmax(attn_weights, dim=-1)                              # (B, L, L)

        # Map attention weights into abstract value space
        # Note the `bmm` method to allow for `head_dim` to be anything
        values = self.V.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, L, D)
        out = torch.bmm(attn_weights, values)                   # (B, L, D)

        return out, attn_weights


class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim, head_dim, context_size, attn_cls, **attn_kwargs):
        """
        Args:
            embed_dim: dimension of input vectors
            head_dim: output dimension of attention head
            context_size: number of tokens in each input
            attn_cls: class of attention head to use (e.g. SelfAttentionHead, RelationalAttentionHead)
            attn_kwargs: extra keyword arguments for the attention head (e.g. linear=True)
        """
        super().__init__()
        self.attn = attn_cls(embed_dim, head_dim, context_size, **attn_kwargs)
        self.fc = nn.Linear(head_dim*context_size, 1)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        out, _ = self.attn(x)              # (B, L, D_h)
        pooled = out.flatten(start_dim=1)  # flatten the tokens together
        logits = self.fc(pooled).squeeze(-1)
        return logits


class QueryKeyScorer(nn.Module):
    def __init__(self, embed_dim, head_dim):
        """
        Minimal model for ordinal relations: the logit for a pair of tokens is
        the (scaled) attention score between the second token's query and the
        first token's key. Separate query and key matrices make the score an
        arbitrary bilinear form x_1^T W_k^T W_q x_2, which can be asymmetric.
        """
        super().__init__()
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)

    def query_key_matrix(self):
        """Bilinear form M with score(x_1, x_2) = x_1^T M x_2 / sqrt(head_dim)."""
        return self.W_k.weight.T @ self.W_q.weight  # (embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch, 2, embed_dim)
        returns: (batch,) logits
        """
        Q = self.W_q(x[:, 1])  # (B, D)
        K = self.W_k(x[:, 0])  # (B, D)
        return (Q * K).sum(dim=-1) / (K.size(-1) ** 0.5)


class CoRelNet(QueryKeyScorer):
    def __init__(self, embed_dim, head_dim):
        """
        Same as QueryKeyScorer, but queries and keys share one matrix W, so the
        score x_1^T W^T W x_2 is symmetric in its two arguments.
        """
        nn.Module.__init__(self)
        self.W = nn.Linear(embed_dim, head_dim, bias=False)

    @property
    def W_q(self):
        return self.W

    @property
    def W_k(self):
        return self.W
