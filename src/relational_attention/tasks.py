import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class BlockComparisonDataset(Dataset):
    def __init__(self, num_samples=1000, num_vectors=2, vector_len=6, seed=42):
        """
        Args:
            num_samples: total number of examples to generate (will be split 50/50 into same/different)
            num_vectors: number of vectors per example
            vector_len: length of each vector (must be even)
            seed: random seed for reproducibility
        """
        assert vector_len % 2 == 0, "vector_len must be even"
        assert num_samples % 2 == 0, "num_samples must be even for balanced classes"

        self.num_samples = num_samples
        self.num_vectors = num_vectors
        self.vector_len = vector_len
        self.mid = vector_len // 2

        # Controlled random generator
        self.rng = np.random.default_rng(seed)

        # Generate data
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        half = self.num_samples // 2
        X_same = [self.generate_same() for _ in range(half)]
        X_diff = [self.generate_different() for _ in range(half)]
        X_all = X_same + X_diff
        y_all = [1.0] * half + [0.0] * half

        # Shuffle the dataset (consistent shuffle across data and labels)
        idxs = self.rng.permutation(self.num_samples)
        X_all = [X_all[i] for i in idxs]
        y_all = [y_all[i] for i in idxs]

        X_tensor = torch.stack([torch.stack(x) for x in X_all])  # (num_samples, num_vectors, vector_len)
        y_tensor = torch.tensor(y_all, dtype=torch.float32)
        return X_tensor, y_tensor

    def generate_same(self):
        same_block = self.rng.choice(["first", "last"])
        sample = []
        for _ in range(self.num_vectors):
            vec = torch.zeros(self.vector_len)
            if same_block == "first":
                idx = self.rng.integers(0, self.mid)
            else:
                idx = self.rng.integers(self.mid, self.vector_len)
            vec[idx] = 1.0
            sample.append(vec)
        return sample

    def generate_different(self):
        sample = []
        for _ in range(self.num_vectors):
            vec = torch.zeros(self.vector_len)
            block = self.rng.choice(["first", "last"])
            if block == "first":
                idx = self.rng.integers(0, self.mid)
            else:
                idx = self.rng.integers(self.mid, self.vector_len)
            vec[idx] = 1.0
            sample.append(vec)

        # Ensure at least one mismatch (force opposite block for first vector)
        if all(torch.sum(v[:self.mid]) == 1 for v in sample) or all(torch.sum(v[self.mid:]) == 1 for v in sample):
            sample[0] = torch.roll(sample[0], shifts=self.mid)
        return sample

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TransitiveInferenceDataset(Dataset):
    def __init__(self, num_samples=1000, num_items=5, train=True, seed=42):
        """
        Transitive inference task:
        Items have a latent order A > B > C > D > E.
        The model sees pairs (x_i, x_j) and must predict whether x_i > x_j (label=1)
        or x_i < x_j (label=0).
        """
        self.num_samples = num_samples
        self.num_items = num_items
        self.train = train
        self.rng = np.random.default_rng(seed)

        # Define latent ordering (A < B < C < D < E)
        self.ranks = torch.arange(num_items).float()

        # Define which pairs to use
        self._generate_pairs()

        # Generate balanced data
        self.data, self.labels = self._generate_data()

    def _generate_pairs(self):
        """Define possible pairs for training and testing."""
        if self.train:
            # Adjacent pairs only
            self.positive_pairs = [(i, i + 1) for i in range(self.num_items - 1)]
            self.negative_pairs = [(i + 1, i) for i in range(self.num_items - 1)]
        else:
            # Skip-one-or-more pairs for transitive inference
            self.positive_pairs = [(i, j) for i in range(self.num_items)
                                   for j in range(self.num_items)
                                   if j > i + 1]  # strictly non-adjacent, forward
            self.negative_pairs = [(j, i) for i in range(self.num_items)
                                   for j in range(self.num_items)
                                   if j > i + 1]  # same but reversed

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


class TransitiveSequenceDataset(Dataset):
    def __init__(self, num_samples=1000, num_items=8, n_seq=5, seed=42):
        """
        Multi-item transitive inference task.
        The model sees a sequence of items and must decide if they are in correct order.

        Args:
            num_samples : total number of examples
            num_items   : total unique symbols (A,B,C,...)
            n_seq       : length of each sequence (e.g. 5 means 5 comparisons)
            seed        : random seed
        """
        self.num_samples = num_samples
        self.num_items = num_items
        self.n_seq = n_seq
        self.rng = np.random.default_rng(seed)
        self.ranks = np.arange(num_items)

        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        half = self.num_samples // 2
        X_pos = [self._generate_ordered() for _ in range(half)]
        X_neg = [self._generate_unordered() for _ in range(half)]
        X_all = X_pos + X_neg
        y_all = [1.0] * half + [0.0] * half

        # consistent shuffle
        idxs = self.rng.permutation(self.num_samples)
        X_all = [X_all[i] for i in idxs]
        y_all = [y_all[i] for i in idxs]

        X_tensor = torch.stack([torch.stack(x) for x in X_all])  # (N, n_seq, num_items)
        y_tensor = torch.tensor(y_all, dtype=torch.float32)
        return X_tensor, y_tensor

    def _generate_ordered(self):
        """Sample a strictly increasing sequence."""
        seq = self.rng.choice(self.num_items, size=self.n_seq, replace=False)
        seq = np.sort(seq)
        return [torch.nn.functional.one_hot(torch.tensor(i), num_classes=self.num_items).float()
                for i in seq]

    def _generate_unordered(self):
        """Sample a sequence that breaks transitivity at least once."""
        while True:
            seq = self.rng.choice(self.num_items, size=self.n_seq, replace=False)
            if not np.all(np.diff(seq) > 0):  # not strictly increasing
                break
        return [torch.nn.functional.one_hot(torch.tensor(i), num_classes=self.num_items).float()
                for i in seq]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class XORDataset(Dataset):
    def __init__(self, num_samples=1000, num_vectors=2, vector_len=6, seed=42):
        """
        XOR-like task:
        Given several one-hot vectors, classify whether they are all identical.

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
