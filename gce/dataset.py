"""
 Contains all the utilities to load the PD dataset
"""
import os
import random
import torch
from torch.utils.data import Subset, TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import numpy as np

from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union


def get_pd_dataloaders(root='./datasets/', batch_size=8):
    """
       Loads the custom diabetes dataset into a dataloader
       :param root: dir of the dataset
       :param batch_size: size of batch
       :return: DataLoader: train_dataloader, DataLoader: test_dataloader.
    """

    datFile = os.path.join(root, "X_y_wo_tremor_10_wofooof.mat")
    mat_file = loadmat(datFile)

    X, y, patient_ids = mat_file['X_cleaned'], mat_file['y_cleaned'].ravel(), mat_file['idx_cleaned'].ravel()

    # Normalize per patient
    X_normalized = np.zeros_like(X, dtype=np.float32)
    unique_patients = np.unique(patient_ids)

    for patient in unique_patients:
        patient_mask = patient_ids == patient
        if np.sum(patient_mask) > 1:  # Need at least 2 samples for normalization
            scaler = StandardScaler()
            X_normalized[patient_mask] = scaler.fit_transform(X[patient_mask])
        else:
            X_normalized[patient_mask] = X[patient_mask]  # Keep as is if only 1 sample

    # train/test split (allowing patient leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, stratify=y
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                                torch.from_numpy(y_test).long())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def get_pd_dataloaders_with_save(root='./datasets/', batch_size=8, save_split=True, save_path=None):
    """
    Loads the PD dataset and optionally saves the train/test split
    """
    datFile = os.path.join(root, "X_y_wo_tremor_10_wfooof_halfamp.mat")
    mat_file = loadmat(datFile)

    X, y, patient_ids = mat_file['X_cleaned'], mat_file['y_cleaned'].ravel(), mat_file['idx_cleaned'].ravel()

    # Normalize per patient
    X_normalized = np.zeros_like(X, dtype=np.float32)
    unique_patients = np.unique(patient_ids)

    for patient in unique_patients:
        patient_mask = patient_ids == patient
        if np.sum(patient_mask) > 1:
            scaler = StandardScaler()
            X_normalized[patient_mask] = scaler.fit_transform(X[patient_mask])
        else:
            X_normalized[patient_mask] = X[patient_mask]

    # train/test split (allowing patient leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, stratify=y
    )

    # Save the split if requested
    if save_split and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_features=X_train.shape[1],
            n_classes=len(np.unique(y))
        )
        print(f"✓ Saved train/test split to {save_path}")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                                 torch.from_numpy(y_test).long())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def get_pd_groupkfold(root='./datasets/', batch_size=16, test_series=0):
    """
    Create train and test dataloaders using GroupKFold - leave one out

    Parameters:
    -----------
    X : numpy array
        Input features
    y : numpy array
        Target labels
    time_series_ids : numpy array
        Unique identifier for each time series
    batch_size : int
        Batch size for dataloaders
    test_series : int
        Which time series to use as test set (0 to 8 for 9 series)
    """
    datFile = os.path.join(root, "pd", "X_y_5_all_features_50gamma_tort.mat")
    mat_file = loadmat(datFile)
    X, y, time_series_ids = mat_file['X_cleaned'], mat_file['y_cleaned'].ravel(), mat_file['idx_cleaned'].ravel()

    # Create GroupKFold splitter with n_splits equal to number of time series
    n_series = len(np.unique(time_series_ids))
    gkf = GroupKFold(n_splits=n_series)

    # Get the specific split we want
    splits = list(gkf.split(X, y, groups=time_series_ids))
    train_idx, test_idx = splits[test_series]

    # Split the data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(),
                                 torch.from_numpy(y_test).long())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def get_pd_full_amplitude_test_data(root='./datasets/', batch_size=16):
    """
    Load full-amplitude data for testing classifier generalization.

    The classifier is already trained on half-amplitude data.
    This loads full-amplitude data (OFF vs ALL frequencies combined) to test generalization.

    :param root: dir of the dataset
    :param batch_size: size of batch
    :return: DataLoader: test_dataloader
    """
    datFile = os.path.join(root, "X_y_all_conditions_85_55_wfooof.mat")
    mat_data = loadmat(datFile)

    frequencies = [125, 110, 85, 55]

    # Collect all ON conditions
    X_on_list, idx_on_list = [], []
    for freq in frequencies:
        X_on_list.append(mat_data[f'X_{freq}_cleaned'])
        idx_on_list.append(mat_data[f'idx_{freq}_cleaned'].flatten())

    X_on = np.vstack(X_on_list)
    idx_on = np.hstack(idx_on_list)
    y_on = np.ones(len(X_on), dtype=int)

    # OFF data
    X_off = mat_data['X_off_cleaned']
    idx_off = mat_data['idx_off_cleaned'].flatten()
    y_off = np.zeros(len(X_off), dtype=int)

    # Combine
    X_all = np.vstack([X_on, X_off])
    y_all = np.hstack([y_on, y_off])
    patient_ids = np.hstack([idx_on, idx_off])

    # Patient-specific normalization
    X_normalized = np.zeros_like(X_all, dtype=np.float32)
    unique_patients = np.unique(patient_ids)

    for patient in unique_patients:
        patient_mask = patient_ids == patient
        if np.sum(patient_mask) > 1:
            scaler = StandardScaler()
            X_normalized[patient_mask] = scaler.fit_transform(X_all[patient_mask])
        else:
            X_normalized[patient_mask] = X_all[patient_mask]

    print(f"\nFull-amplitude TEST data loaded:")
    print(f"  Total samples: {len(X_normalized)}")
    print(f"  Patients: {sorted(unique_patients)}")
    print(f"  ON samples: {len(X_on)} (across 4 frequencies)")
    for freq in frequencies:
        print(f"    {freq}Hz: {len(mat_data[f'X_{freq}_cleaned'])} samples")
    print(f"  OFF samples: {len(X_off)}")
    print(
        f"  Class balance: {len(y_on) / (len(y_on) + len(y_off)) * 100:.1f}% ON, {len(y_off) / (len(y_on) + len(y_off)) * 100:.1f}% OFF\n")

    test_dataset = TensorDataset(torch.from_numpy(X_normalized).float(),
                                 torch.from_numpy(y_all).long())

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader


def get_indices(dataset):
    """
    Returns indices of datapoint that have a non-negative label.
    :param dataset: dataset to get indices from
    :return: list: indices
    """
    indices = []
    for i, (x, y) in enumerate(dataset):
        if y != -1:
            indices.append(i)

    return indices
