"""
The CIFAR-10 dataset from the torchvision package.
"""

from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base
import torch

class DataSource(base.DataSource):
    """The CIFAR-10 dataset."""

    def __init__(self):
        super().__init__()
        _path = Config().params['data_path']

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
                                       ])

        train_dataset = datasets.CIFAR10(root=_path,
                                         train=True,
                                         download=True,
                                         transform=transform)
        self.testset = datasets.CIFAR10(root=_path,
                                        train=False,
                                        download=True,
                                        transform=transform)

        train_len = int(len(train_dataset) * 0.8)
        validation_len = int(len(train_dataset) - train_len)

        self.trainset, self.validationset = torch.utils.data.random_split(train_dataset,
                                                                          [train_len, validation_len],
                                                                          generator=torch.Generator().manual_seed(
                                                                              42))
