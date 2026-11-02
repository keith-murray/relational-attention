import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from relational_attention.tasks import OrdinalRelationsDataset
from relational_attention.models import QueryKeyScorer, CoRelNet
from relational_attention.training import train_model, evaluate_model

# Task Parameters
num_samples = 1000
num_items = 10

# Model parameters: CoRelNet gets more capacity to show capacity is not the problem
head_dims = {'query_key': 2, 'corelnet': num_items}

# Training parameters
lr = 1e-3
loss_fn = torch.nn.BCEWithLogitsLoss()
epochs = 500
num_models = 30

# Testing parameters
save_loc = './data/'


def create_task():
    # Train on every pair: this tests what each model can represent, not generalization
    dataset = OrdinalRelationsDataset(num_samples=num_samples, num_items=num_items, pairs="all")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    return loader


def initialize_models():
    model_query_key = QueryKeyScorer(embed_dim=num_items, head_dim=head_dims['query_key'])
    model_corelnet = CoRelNet(embed_dim=num_items, head_dim=head_dims['corelnet'])

    return {'query_key': model_query_key, 'corelnet': model_corelnet}


def antisymmetry(model):
    """Fraction of the query-key matrix's norm in its antisymmetric part."""
    M = model.query_key_matrix().detach()
    return (torch.linalg.norm(M - M.T) / (torch.linalg.norm(M - M.T) + torch.linalg.norm(M + M.T))).item()


if __name__ == "__main__":
    results = []

    for seed in range(num_models):
        print(f'seed {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)

        loader_train = create_task()
        models = initialize_models()

        for name, model in models.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss = train_model(model, loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs, verbose=False)

            results.append({
                'seed': seed,
                'model': name,
                'final_loss': loss[-1],
                'accuracy': evaluate_model(model, loader_train),
                'antisymmetry': antisymmetry(model),
            })

    df = pd.DataFrame(results)
    df.to_csv(save_loc + 'ordinal_relations_results.csv', index=False)
    print(df.groupby('model')[['final_loss', 'accuracy', 'antisymmetry']].describe().T)
