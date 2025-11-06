import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from relational_attention.tasks import XORDataset
from relational_attention.models import SelfAttentionHead, RelationalAttentionHead, AttentionClassifier
from relational_attention.training import train_model

# Task Parameters
num_samples = 1000
num_vectors=2
vector_len=2

# Training parameters
lr = 1e-3
loss_fn = torch.nn.BCEWithLogitsLoss()
epochs = 250
num_models = 30

# Testing parameters
save_loc = './data/'


def create_task():
    dataset = XORDataset(
        num_samples=num_samples,
        num_vectors=num_vectors,
        vector_len=vector_len,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    return loader


def initialize_models():
    # Self attention head with softmax
    model_self_attn = AttentionClassifier(
        embed_dim=vector_len, head_dim=vector_len, context_size=num_vectors, attn_cls=SelfAttentionHead
    )

    # Relational attention head with softmax
    model_relat_attn = AttentionClassifier(
        embed_dim=vector_len, head_dim=vector_len, context_size=num_vectors, attn_cls=RelationalAttentionHead
    )

    return model_self_attn, model_relat_attn


def train_models_on_xor(model_self, model_relat, loader_train):
    optimizer = torch.optim.Adam(model_self.parameters(), lr=lr)
    loss_self_attn = train_model(model_self, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs,)

    optimizer = torch.optim.Adam(model_relat.parameters(), lr=lr)
    loss_relat_attn = train_model(model_relat, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs,)

    return loss_self_attn, loss_relat_attn


if __name__ == "__main__":
    results = []

    for seed in range(num_models):
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader_train = create_task()
        model_self, model_relat = initialize_models()

        loss_self_attn, loss_relat_attn = train_models_on_xor(model_self, model_relat, loader_train)

        for epoch, (l_self, l_relat) in enumerate(zip(loss_self_attn, loss_relat_attn)):
            results.append({
                'seed': seed,
                'epoch': epoch,
                'self_attention_loss': l_self,
                'relational_attention_loss': l_relat
            })

    df = pd.DataFrame(results)
    df.to_csv(save_loc + 'xor_results.csv', index=False)
    print(df)