import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from relational_attention.tasks import XORDataset
from relational_attention.models import SelfAttentionHead, RelationalAttentionHead, AttentionClassifier
from relational_attention.training import train_model

# Task Parameters
num_samples = 1000
num_vectors = 2
vector_len = 2

# Training parameters
lr = 1e-3
loss_fn = torch.nn.BCEWithLogitsLoss()
epochs = 250
num_models = 30

# Testing parameters
save_loc = './data/'

# Attention head variants: (name, attention class, linear flag)
variants = [
    ('self_attention', SelfAttentionHead, False),
    ('relational_attention', RelationalAttentionHead, False),
    ('self_attention_linear', SelfAttentionHead, True),
    ('relational_attention_linear', RelationalAttentionHead, True),
]


def create_task():
    dataset = XORDataset(
        num_samples=num_samples,
        num_vectors=num_vectors,
        vector_len=vector_len,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    return loader


def initialize_model(attn_cls, linear):
    return AttentionClassifier(
        embed_dim=vector_len, head_dim=vector_len, context_size=num_vectors, attn_cls=attn_cls, linear=linear
    )


def train_model_on_xor(model, loader_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return train_model(model, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs, verbose=False)


if __name__ == "__main__":
    results = {}

    for seed in range(num_models):
        print(f'seed {seed}')
        for name, attn_cls, linear in variants:
            torch.manual_seed(seed)
            np.random.seed(seed)

            loader_train = create_task()
            model = initialize_model(attn_cls, linear)

            for epoch, loss in enumerate(train_model_on_xor(model, loader_train)):
                results.setdefault((seed, epoch), {'seed': seed, 'epoch': epoch})[f'{name}_loss'] = loss

    df = pd.DataFrame(list(results.values()))
    df.to_csv(save_loc + 'xor_results.csv', index=False)
    print(df.groupby('epoch').mean().tail(1))
