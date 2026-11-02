import torch
from torch.utils.data import Dataset
import numpy as np


class XORDataset(Dataset):
    def __init__(self, num_samples=1000, num_vectors=2, vector_len=6, seed=42):
        """
        XOR-like task:
        Given several one-hot vectors, classify whether they are all identical
        (label=0) or not all identical (label=1). With two 2-dimensional
        vectors this is exactly XOR.

        Args:
            num_samples : total number of examples (balanced 50/50 same/different)
            num_vectors : number of vectors per example
            vector_len  : dimensionality of each one-hot vector
            seed        : random seed for reproducibility
        """
        assert num_samples % 2 == 0, "num_samples must be even for balanced classes"

        self.num_samples = num_samples
        self.num_vectors = num_vectors
        self.vector_len = vector_len
        self.rng = np.random.default_rng(seed)

        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        half = self.num_samples // 2
        X_same = [self._generate_same() for _ in range(half)]
        X_diff = [self._generate_different() for _ in range(half)]
        X_all = X_same + X_diff
        y_all = [0.0] * half + [1.0] * half

        # Consistent shuffle
        idxs = self.rng.permutation(self.num_samples)
        X_all = [X_all[i] for i in idxs]
        y_all = [y_all[i] for i in idxs]

        X_tensor = torch.stack([torch.stack(x) for x in X_all])  # (num_samples, num_vectors, vector_len)
        y_tensor = torch.tensor(y_all, dtype=torch.float32)
        return X_tensor, y_tensor

    def _generate_same(self):
        """All vectors are identical (same one-hot index)."""
        idx = self.rng.integers(0, self.vector_len)
        vec = torch.nn.functional.one_hot(torch.tensor(idx), num_classes=self.vector_len).float()
        return [vec.clone() for _ in range(self.num_vectors)]

    def _generate_different(self):
        """At least one vector differs from the others."""
        indices = self.rng.choice(self.vector_len, size=self.num_vectors, replace=False)
        vectors = [torch.nn.functional.one_hot(torch.tensor(i), num_classes=self.vector_len).float()
                   for i in indices]
        return vectors

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class OrdinalRelationsDataset(Dataset):
    def __init__(self, num_samples=1000, num_items=5, pairs="all", seed=42):
        """
        Ordinal relations task:
        Items have a latent order A < B < C < D < E. The model sees a pair of
        one-hot items (x_i, x_j) and must predict whether x_i < x_j (label=1)
        or x_i > x_j (label=0).

        Args:
            num_samples : total number of examples (balanced 50/50)
            num_items   : number of items in the latent order
            pairs       : which pairs to sample from
                          "all"         - every pair of distinct items
                          "adjacent"    - only neighbouring items (e.g. B,C)
                          "nonadjacent" - only items at least two ranks apart
            seed        : random seed for reproducibility
        """
        assert pairs in ("all", "adjacent", "nonadjacent"), f"unknown pairs: {pairs}"

        self.num_samples = num_samples
        self.num_items = num_items
        self.pairs = pairs
        self.rng = np.random.default_rng(seed)

        # Define latent ordering (A < B < C < D < E)
        self.ranks = torch.arange(num_items).float()

        # Define which pairs to use
        self._generate_pairs()

        # Generate balanced data
        self.data, self.labels = self._generate_data()

    def _generate_pairs(self):
        """Define the forward (label=1) pairs and their reversals (label=0)."""
        min_gap = {"all": 1, "adjacent": 1, "nonadjacent": 2}[self.pairs]
        max_gap = 1 if self.pairs == "adjacent" else self.num_items

        self.positive_pairs = [(i, j) for i in range(self.num_items)
                               for j in range(self.num_items)
                               if min_gap <= j - i <= max_gap]  # forward
        self.negative_pairs = [(j, i) for i, j in self.positive_pairs]  # reversed

    def _generate_data(self):
        """Generate a balanced dataset of (pair, label) samples."""
        half = self.num_samples // 2
        X_pos = [self._make_sample(self.positive_pairs, label=1.0) for _ in range(half)]
        X_neg = [self._make_sample(self.negative_pairs, label=0.0) for _ in range(half)]
        X_all = X_pos + X_neg
        y_all = [1.0] * half + [0.0] * half

        # Shuffle consistently
        idxs = self.rng.permutation(self.num_samples)
        X_all = [X_all[i] for i in idxs]
        y_all = [y_all[i] for i in idxs]

        X_tensor = torch.stack([torch.stack(x) for x in X_all])  # (num_samples, 2, num_items)
        y_tensor = torch.tensor(y_all, dtype=torch.float32)
        return X_tensor, y_tensor

    def _make_sample(self, pair_list, label):
        """Pick a pair and convert to one-hot vectors."""
        i, j = pair_list[self.rng.integers(0, len(pair_list))]
        x_i = torch.nn.functional.one_hot(torch.tensor(i), num_classes=self.num_items).float()
        x_j = torch.nn.functional.one_hot(torch.tensor(j), num_classes=self.num_items).float()
        return [x_i, x_j]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
