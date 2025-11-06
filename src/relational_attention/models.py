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
            attn_cls: class of attention head to use (e.g. SimpleAttentionHead, AbstractAttentionHead)
            attn_kwargs: extra keyword arguments for the attention head (e.g. num_values)
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
