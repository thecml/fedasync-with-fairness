from typing import Optional, Callable, Tuple, Any

from torchvision import datasets, transforms
import numpy as np
from PIL import Image as Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms
from wandb.wandb_torch import torch

from plato.config import Config
from plato.datasources import base
import os
from pathlib import Path

class CustomCifar10(VisionDataset):
    def __init__(
            self, data, targets, transform: Optional[Callable] = transforms.ToTensor(),
            target_transform: Optional[Callable] = None) \
            -> None:
        super().__init__(root=None, transform=transform, target_transform=target_transform)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
        self.classes = range(10)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class DataSource(base.DataSource):
    def __init__(self, client_id=0):
        super().__init__()
        self.trainset = None
        self.testset = None
        self.validationset = None
        self.client_sizes = dict()

        curr_dir = os.getcwd()
        root_dir = Path(curr_dir).absolute()
        data_dir = Path.joinpath(root_dir, "data/partitioned_datasets/cifar10/lda/concentration_100")

        self._root = data_dir

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
        ])
        
        # Save size of clients
        n_clients = 10
        for client_id in range(1, n_clients+1):
            partition_folder = f"{self._root}/Client-{client_id}"
            client_data = np.load(f"{partition_folder}/train_samples.npy")
            self.client_sizes[client_id] = client_data.shape[0]
            
        # Server is retrieving the data with client_id=0. Therefore giving all data. Even though it only uses test
        if client_id == 0:
            train_samples = []
            train_targets = []
            val_samples = []
            val_targets = []
            test_samples = []
            test_targets = []

            for i in range(n_clients):
                partition_folder = f"{self._root}/Client-{i + 1}"
                train_samples.append(np.load(f"{partition_folder}/train_samples.npy"))
                train_targets.append(np.load(f"{partition_folder}/train_labels.npy"))
                val_samples.append(np.load(f"{partition_folder}/val_samples.npy"))
                val_targets.append(np.load(f"{partition_folder}/val_labels.npy"))
                test_samples.append(np.load(f"{partition_folder}/test_samples.npy"))
                test_targets.append(np.load(f"{partition_folder}/test_labels.npy"))

            self.trainset = CustomCifar10(np.concatenate(train_samples), np.concatenate(train_targets), _transform)
            self.validationset = CustomCifar10(np.concatenate(val_samples), np.concatenate(val_targets), _transform)
            self.testset = CustomCifar10(np.concatenate(test_samples), np.concatenate(test_targets), _transform)
            return

        # Specific partition folder
        partition_folder = f"{self._root}/Client-{client_id}"

        # Load Train data
        train_samples = np.load(f"{partition_folder}/train_samples.npy")
        train_targets = np.load(f"{partition_folder}/train_labels.npy")
        self.trainset = CustomCifar10(train_samples, train_targets, _transform)

        # Load val data
        val_samples = np.load(f"{partition_folder}/val_samples.npy")
        val_targets = np.load(f"{partition_folder}/val_labels.npy")
        self.validationset = CustomCifar10(val_samples, val_targets, _transform)

        # Load test data
        test_samples = np.load(f"{partition_folder}/test_samples.npy")
        test_targets = np.load(f"{partition_folder}/test_labels.npy")
        self.testset = CustomCifar10(test_samples, test_targets, _transform)


