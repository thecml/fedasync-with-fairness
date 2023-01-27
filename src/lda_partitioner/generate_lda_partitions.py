'''
This file was copied and then modified from an ealier project by master thesis students Anton and Peter.
'''
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import os
from sklearn.model_selection import train_test_split

from numpy import save
from numpy import asarray
from numpy import load

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras.datasets import mnist, cifar10

from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters, Scalar, NDArrays
from flower_baselines.flwr_baselines.dataset.utils.common import (
    XY,
    create_lda_partitions,
    shuffle,
    sort_by_label,
    split_array_at_indices,
)

def gen_dataset_partitions(
    path,
    dataset_name,
    num_total_clients,
    lda_concentrations = [10],
):
    """Defines root path for partitions and calls functions to create them."""

    fed_dir = f"{path}"
    print(fed_dir)

    train_x, train_y, val_x, val_y, test_x, test_y = get_base_dataset(dataset_name)
    print(f"test : {len(test_y)}")
    print(f"val : {len(val_y)}")
    print(f"train : {len(train_y)}")

    for concentration in lda_concentrations:
        # partion LDA train
        concentration_dirname = ("/concentration_{}".format(concentration)).replace(".", "")
        dist = partition_dataset_and_save(
            dataset=(train_x, train_y),
            fed_dir=fed_dir+concentration_dirname,
            dirichlet_dist=None,
            num_partitions=num_total_clients,
            concentration=concentration,
            partition_type="train",
        )

        # Use dist distribution 'dist' from train generation
        # partion LDA val
        partition_dataset_and_save(
            dataset=(val_x, val_y),
            fed_dir=fed_dir+concentration_dirname,
            dirichlet_dist=dist,
            num_partitions=num_total_clients,
            concentration=concentration,
            partition_type="val",
        )

        # test
        # partion LDA test
        partition_dataset_and_save(
            dataset=(test_x, test_y),
            fed_dir=fed_dir+concentration_dirname,
            dirichlet_dist=dist,
            num_partitions=num_total_clients,
            concentration=concentration,
            partition_type="test",
        )

    return fed_dir

def get_base_dataset(dataset_name):
    if dataset_name == "MNIST":
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        print(f"total : {len(train_y)+len(test_y)}")
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.16666666666, random_state=42)
    else:
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        print(f"total : {len(train_y)+len(test_y)}")
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    return train_x, train_y, val_x, val_y, test_x, test_y

def partition_dataset_and_save(
    dataset,
    fed_dir,
    dirichlet_dist=None,
    num_partitions=500,
    concentration=0.1,
    partition_type="train",
):
    """Creates and saves partitions for CIFAR10.
    Args:
        dataset (XY): Original complete dataset.
        fed_dir (Path): Root directory where to save partitions.
        dirichlet_dist (Optional[npt.NDArray[np.float32]], optional):
            Pre-defined distributions to be used for sampling if exist. Defaults to None.
        num_partitions (int, optional): Number of partitions. Defaults to 500.
        concentration (float, optional): Alpha value for Dirichlet. Defaults to 0.1.
        partition_type: string
    Returns:
        np.ndarray: Generated dirichlet distributions.
    """
    # Create partitions
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=concentration,
        accept_imbalanced=True,
        seed=69
    )
    print(clients_partitions[0][0].shape)

    # Save partions
    save_partitions(list_partitions=clients_partitions, fed_dir=fed_dir, partition_type=partition_type)

    return dist

def save_partitions(list_partitions, fed_dir, partition_type="train"):
    """Saves partitions to individual files.
    Args:
        list_partitions (List[XY]): List of partitions to be saves
        fed_dir (Path): Root directory where to save partitions.
        partition_type (str, optional): Partition type ("train" or "test"). Defaults to "train".
    """
    for idx, partition in enumerate(list_partitions):
        path_dir = f"{fed_dir}/Client-{idx+1}"
        os.makedirs(path_dir, exist_ok=True)
        save(f"{path_dir}/{partition_type}_samples.npy", partition[0])
        save(f"{path_dir}/{partition_type}_labels.npy", partition[1])

if __name__ == "__main__":
    # Make partitioned datasets
    curr_dir = os.getcwd()
    root_dir = Path(curr_dir).absolute()
    mnist_dir = Path.joinpath(root_dir, "data/partitioned_datasets/mnist/lda/")
    cifar10_dir = Path.joinpath(root_dir, "data/partitioned_datasets/cifar10/lda/")
    gen_dataset_partitions(mnist_dir, "MNIST", 10, [1, 10, 100, 1000])
    gen_dataset_partitions(cifar10_dir, "CIFAR10", 10, [1, 10, 100, 1000])