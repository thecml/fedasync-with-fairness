"""
The MNIST dataset from the torchvision package.
"""
from torchvision import datasets, transforms
import torch
from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """ The MNIST dataset. """

    def __init__(self):
        super().__init__()
        _path = Config().params['data_path']

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        train_dataset = datasets.MNIST(root=_path,
                                       train=True,
                                       download=True,
                                       transform=_transform)
        self.testset = datasets.MNIST(root=_path,
                                      train=False,
                                      download=True,
                                      transform=_transform)

        train_len = int(len(train_dataset) * 0.8)
        validation_len = int(len(train_dataset) - train_len)

        self.trainset, self.validationset = torch.utils.data.random_split(train_dataset,
                                                                               [train_len, validation_len],
                                                                               generator=torch.Generator().manual_seed(
                                                                                   42))
