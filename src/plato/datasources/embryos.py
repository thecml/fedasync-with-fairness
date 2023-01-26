import enum as enum
from plato.datasources import base
import os
import pandas as pd
from PIL import Image as Image
from typing import Callable, Tuple, Any, Optional
from torchvision.datasets import VisionDataset
import numpy as np
import torch
import numpy
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from plato.config import Config
import logging

class EmbryosDataset(VisionDataset):

    def __init__(
        self, data, targets, clinic_ids, root = "", transform: Optional[Callable] = transforms.ToTensor(), target_transform: Optional[Callable] = None) \
            -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
        self.clinic_ids = clinic_ids
        self._root = root
        self.classes = [0, 1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        sampleId = self.data.iloc[[index]]['SampleID'].values[0]
        file_path = os.path.join(self._root, "{:05d}.npz".format(sampleId))
        img, target = self._load_image(file_path), int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    
    def _load_image(self, path):
        file_data = np.load(path)
        images = file_data['images']

        focal = 1
        frame = 0
        img_raw = images[frame, :, :, focal]
        img = Image.fromarray(img_raw)
        newsize = (250, 250)
        img = img.resize(newsize)
        img_raw = np.asarray(img)
        img_raw = img_raw.astype('float32') / 255
        return img_raw

class DataSource(base.DataSource):
    def __init__(self, client_id=0):
        super().__init__()
        self._oversampling = False
        self.trainset = None
        self.testset = None
        self.validationset = None
        self._root = "/mnt/data/mlr_ahj_datasets/vitrolife/dataset/"

        #Loading in the meta data file
        metadata_file_path = os.path.join(self._root, "metadata.csv")
        self._meta_data = pd.read_csv(metadata_file_path)
        
        # If indicated in the config file - The clients will be using the labIds in the order from most data to least, meaning that when using a lower amount of clients than labIds the labs with the lowest amount of training data are not used
        sortedIds = self._size_sort_client_ids() if (hasattr(Config().data, "size_sorted") and Config().data.size_sorted) else range(23)
        if client_id != 0: logging.info("client-id #%d will be designated labId #%d", client_id, sortedIds[client_id-1])
        
        # Splitting train, test and validation data
        meta_data_train_validation = self._meta_data.loc[self._meta_data['Testset'] == 0]
        meta_data_test = self._meta_data.loc[self._meta_data['Testset'] == 1]
        
        if client_id != 0:
            client_train_validation_data = meta_data_train_validation.loc[meta_data_train_validation['LabID'] == sortedIds[client_id-1]]
            client_train_data, client_validation_data = train_test_split(client_train_validation_data, test_size=0.172, random_state=42)
            client_test_data = meta_data_test.loc[meta_data_test['LabID'] == sortedIds[client_id-1]]
        else:
            client_train_data, client_validation_data = train_test_split(meta_data_train_validation, test_size=0.172, random_state=42)
            client_test_data = meta_data_test

        #Loading in data
        train_data, train_targets, train_clinic_ids = self._load_data_type(client_train_data)
        validation_data, validation_targets, validation_clinic_ids = self._load_data_type(client_validation_data)
        test_data, test_targets, test_clinic_ids = self._load_data_type(client_test_data)

        self.trainset = EmbryosDataset(data=train_data, targets=train_targets, clinic_ids=train_clinic_ids, root=self._root)
        self.validationset = EmbryosDataset(data=validation_data, targets=validation_targets, clinic_ids=validation_clinic_ids, root=self._root)
        self.testset = EmbryosDataset(data=test_data, targets=test_targets, clinic_ids=test_clinic_ids, root=self._root)

    def _size_sort_client_ids(self):
        logging.info("Sorting Embryo dataset based on lab size")
        ids = range(23)
        sizes = []
        train_val_data = self._meta_data.loc[self._meta_data['Testset'] == 0]
        for id in ids:
            sizes.append(len(train_val_data.loc[train_val_data['LabID'] == id]))
        assert(len(sizes) == 23)
        sortedIds = [x for _,x in sorted(zip(sizes,ids), reverse=True)]
        self.sortedIds = sortedIds
        return sortedIds



    def _load_data_type(self, metadata):
        data = metadata
        labels = data["Label"].tolist()
        labels = torch.LongTensor(labels)
        clinic_ids = data["LabID"].tolist()
        clinic_ids = torch.LongTensor(clinic_ids)
        
        return data, labels, clinic_ids