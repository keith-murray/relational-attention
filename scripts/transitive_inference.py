import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from relational_attention.tasks import TransitiveInferenceDataset
from relational_attention.models import SelfAttentionHead, RelationalAttentionHead, AttentionClassifier
from relational_attention.training import train_model, evaluate_model

# Task Parameters
num_samples = 1000
num_items = 5

# Training parameters
lr = 1e-3
loss_fn = torch.nn.BCEWithLogitsLoss()
epochs = 250
num_models = 30

# Testing parameters
save_loc = './data/'


def create_task():
    # Training task
    dataset_train = TransitiveInferenceDataset(num_samples=num_samples, num_items=num_items)
    loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

    # Testing task
    dataset_test = TransitiveInferenceDataset(num_samples=num_samples, num_items=num_items, train=False)
    loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

    return loader_train, loader_test


def initialize_models():
    # Self attention head with softmax
    model_self_attn = AttentionClassifier(
        embed_dim=num_items, head_dim=num_items, context_size=2, attn_cls=SelfAttentionHead
    )

    # Linear self attention head
    model_linear_self = AttentionClassifier(
        embed_dim=num_items, head_dim=num_items, context_size=2, attn_cls=SelfAttentionHead, linear=True
    )

    # Relational attention head with softmax
    model_relat_attn = AttentionClassifier(
        embed_dim=num_items, head_dim=num_items, context_size=2, attn_cls=RelationalAttentionHead
    )

    # Linear relational attention head
    model_linear_relat = AttentionClassifier(
        embed_dim=num_items, head_dim=num_items, context_size=2, attn_cls=RelationalAttentionHead, linear=True
    )

    return model_self_attn, model_relat_attn, model_linear_self, model_linear_relat


def train_models_on_paired_ti(model_self_attn, model_relat_attn, model_linear_self, model_linear_relat, loader_train):
    optimizer = torch.optim.Adam(model_self_attn.parameters(), lr=lr)
    _ = train_model(model_self_attn, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs,)

    optimizer = torch.optim.Adam(model_relat_attn.parameters(), lr=lr)
    _ = train_model(model_relat_attn, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs,)

    optimizer = torch.optim.Adam(model_linear_self.parameters(), lr=lr)
    _ = train_model(model_linear_self, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs,)

    optimizer = torch.optim.Adam(model_linear_relat.parameters(), lr=lr)
    _ = train_model(model_linear_relat, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs,)

    return None


def test_models_on_ti(model_self_attn, model_relat_attn, model_linear_self, model_linear_relat, loader_test):
    acc_self = evaluate_model(model_self_attn, loader_test)
    acc_relat = evaluate_model(model_relat_attn, loader_test)
    acc_self_linear = evaluate_model(model_linear_self, loader_test)
    acc_relat_linear = evaluate_model(model_linear_relat, loader_test)

    return acc_self, acc_relat, acc_self_linear, acc_relat_linear


if __name__ == "__main__":
    results = []

    for seed in range(num_models):
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader_train, loader_test = create_task()
        model_self_attn, model_relat_attn, model_linear_self, model_linear_relat = initialize_models()

        train_models_on_paired_ti(model_self_attn, model_relat_attn, model_linear_self, model_linear_relat, loader_train)
        acc_self, acc_relat, acc_self_linear, acc_relat_linear = test_models_on_ti(
            model_self_attn, model_relat_attn, model_linear_self, model_linear_relat, loader_test
        )

        results.append({
            'seed': seed,
            'self_attention_acc': acc_self,
            'relational_attention_acc': acc_relat,
            'self_attention_linear_acc': acc_self_linear,
            'relational_attention_linear_acc': acc_relat_linear,
        })

    df = pd.DataFrame(results)
    df.to_csv(save_loc + 'paired_ti_results.csv', index=False)
    print(df)