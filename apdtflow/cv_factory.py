"""
This module defines a TimeSeriesCVFactory class that can generate various types of cross‑validation
splits for time series data. This “factory” allows you to choose between different CV techniques
(such as rolling window, expanding window, or blocked splits).
"""


class TimeSeriesCVFactory:
    def __init__(
        self, dataset, method="rolling", train_size=None, val_size=None, step_size=None
    ):
        """
        Initialize the cross-validation factory.

        Args:
            dataset (Dataset): The time series dataset.
            method (str): The CV method to use ("rolling", "expanding", "blocked").
            train_size (int): Number of samples for training.
            val_size (int): Number of samples for validation.
            step_size (int): How many samples to move forward in each split.
        """
        allowed_methods = {"rolling", "expanding", "blocked"}
        if method not in allowed_methods:
            raise ValueError(f"Cross-validation method '{method}' is not supported.")
        self.dataset = dataset
        self.method = method
        self.train_size = train_size
        self.val_size = val_size
        self.step_size = step_size
        self.n_samples = len(dataset)

    def rolling_window_splits(self, max_splits=None):
        """
        Generate splits using a rolling window strategy.
        """
        splits = []
        start = 0
        count = 0
        while start + self.train_size + self.val_size <= self.n_samples:
            train_indices = list(range(start, start + self.train_size))
            val_indices = list(
                range(start + self.train_size, start + self.train_size + self.val_size)
            )
            splits.append((train_indices, val_indices))
            start += self.step_size
            count += 1
            if max_splits is not None and count >= max_splits:
                break
        return splits

    def expanding_window_splits(self, max_splits=None):
        """
        Generate splits using an expanding window strategy.
        """
        splits = []
        end_train = self.train_size
        count = 0
        while end_train + self.val_size <= self.n_samples:
            train_indices = list(range(0, end_train))
            val_indices = list(range(end_train, end_train + self.val_size))
            splits.append((train_indices, val_indices))
            end_train += self.step_size
            count += 1
            if max_splits is not None and count >= max_splits:
                break
        return splits

    def blocked_splits(self):
        """
        Generate a single blocked split where the dataset is divided into training and validation parts.
        """
        if self.train_size is None or self.val_size is None:
            raise ValueError(
                "For blocked splits, train_size and val_size must be provided."
            )
        train_indices = list(range(0, self.train_size))
        val_indices = list(range(self.train_size, self.train_size + self.val_size))
        return [(train_indices, val_indices)]

    def get_splits(self, max_splits=None):
        """
        Get the cross validation splits based on the chosen method.
        """
        if self.method == "rolling":
            return self.rolling_window_splits(max_splits)
        elif self.method == "expanding":
            return self.expanding_window_splits(max_splits)
        elif self.method == "blocked":
            return self.blocked_splits()
        else:
            raise ValueError(
                f"Cross-validation method '{self.method}' is not supported."
            )
