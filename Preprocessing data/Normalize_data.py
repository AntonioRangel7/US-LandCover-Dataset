"""
Authors: Antonio Rangel & Juan Terven
Date: 2023

Description:
This script computes dataset statistics (mean and standard deviation) and normalizes image data stored in HDF5 format.
The normalization process ensures that the dataset follows a standard distribution, improving model performance and training stability.

Main Features:
- Computes per-channel mean and standard deviation for large datasets in an efficient batch-wise manner.
- Implements a normalization module compatible with PyTorch (`MyNormalize`).
- Applies normalization to datasets stored in HDF5 format and saves results to a new file.
- Supports configurable batch sizes and dataset percentage for memory-efficient processing.
- Provides a command-line interface for user-defined dataset paths and normalization parameters.
"""


import h5py
import torch
from typing import List
from tqdm import tqdm
import numpy as np


def save_batch_to_hdf5(file_name, masks, predictors, append=False):
    """
    Saves batches of mask and predictor data to an HDF5 file.

    Args:
        file_name (str): The HDF5 file path where data will be stored.
        masks (np.ndarray): Array of mask tiles with shape (num_tiles, 256, 256).
        predictors (np.ndarray): Array of predictor tiles with shape (num_tiles, 3, 256, 256).
        append (bool): If True, appends data to an existing file; otherwise, overwrites it.
    """
    with h5py.File(file_name, 'a') as f:
        if 'masks' not in f or not append:
            f.create_dataset(
                'masks',
                data=masks,
                maxshape=(None, masks.shape[1], masks.shape[2]),
                chunks=True,
                compression="gzip"
            )
            f.create_dataset(
                'predictors',
                data=predictors,
                maxshape=(None, predictors.shape[1], predictors.shape[2], predictors.shape[3]),
                chunks=True,
                compression="gzip"
            )
        else:
            f['masks'].resize((f['masks'].shape[0] + masks.shape[0]), axis=0)
            f['masks'][-masks.shape[0]:] = masks

            f['predictors'].resize((f['predictors'].shape[0] + predictors.shape[0]), axis=0)
            f['predictors'][-predictors.shape[0]:] = predictors

def calc_statistics_hdf5(file_name, batch_size, porcentage_dataset=1):
    """
    Calculate the statistics (mean and std) for the entire HDF5 dataset.
    The calculation is performed in batches to handle large datasets efficiently.
    """
    accum_mean = None
    accum_std = None
    num_batches = 0

    with h5py.File(file_name, 'r') as f:
        total_elements = int(f['masks'].shape[0] * porcentage_dataset)

        for start in tqdm(range(0, total_elements, batch_size), desc="Processing batches"):
            end = min(start + batch_size, total_elements)
            predictors_batch = f['predictors'][start:end]
            predictors_flat = predictors_batch.reshape((predictors_batch.shape[0], predictors_batch.shape[1], -1))

            batch_mean = predictors_flat.mean(axis=2).sum(axis=0)
            batch_std = predictors_flat.std(axis=2).sum(axis=0)

            if accum_mean is None:
                accum_mean = batch_mean
                accum_std = batch_std
            else:
                accum_mean += batch_mean
                accum_std += batch_std

            num_batches += predictors_batch.shape[0]

    final_mean = accum_mean / num_batches
    final_std = accum_std / num_batches
    return final_mean, final_std

class MyNormalize(torch.nn.Module):
    def __init__(self, mean: List[float], stdev: List[float], batch_size: int = 32, porcentage_dataset=1):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(stdev).view(-1, 1, 1)
        self.batch_size = batch_size
        self.porcentage_dataset = porcentage_dataset

    def forward(self, x):
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)

        if x.size(1) != mean.size(0):
            raise ValueError(
                f"Mismatch between input channels ({x.size(1)}) and normalization parameters ({mean.size(0)})."
            )
        return (x - mean) / std

    def normalize_hdf5(self, input_file: str, output_file: str):
        """
        Normalize the data stored in an HDF5 file and save to another file.
        """
        with h5py.File(input_file, "r") as h5f_in:
            dataset_in = h5f_in["predictors"]
            masks = h5f_in["masks"]
            self._process_dataset_in_batches(dataset_in, output_file, masks)

    def _process_dataset_in_batches(self, dataset_in, dataset_out, masks):
        total_samples = int(dataset_in.shape[0] * self.porcentage_dataset)
        for i in tqdm(range(0, total_samples, self.batch_size), desc="Normalizing batches"):
            start = i
            end = i + self.batch_size
            predictors = torch.tensor(dataset_in[start:end])
            masks_batch = masks[start:end]
            normalized_batch = self.forward(predictors)
            save_batch_to_hdf5(dataset_out, masks_batch, normalized_batch.numpy(), append=True)


def ask_user(message="Enter a number: "):
    """
    Prompts the user with a given message and returns the input.

    Args:
        message (str): The message to display to the user.

    Returns:
        str: The user's input as a string.
    """
    return input(message)


def main():
    """
    Main function to handle user input, compute dataset statistics, and normalize training and validation data.

    This script prompts the user to provide paths for training and validation datasets, computes
    mean and standard deviation from the training dataset, and normalizes both the training and validation datasets
    using the computed statistics.
    """

    # Prompt the user for input paths
    training_data_path = ask_user(message="Enter the path for your training data: ")
    validation_data_path = ask_user(message="Enter the path for your validation data: ")
    output_directory = ask_user(message="Enter the output directory: ")

    # Ask the user for the dataset percentage to be used
    dataset_percentage = float(ask_user(message="Enter the percentage of the dataset to use (e.g., 1.0 for 100%): "))

    # Compute mean and standard deviation from the training dataset
    h5_mean, h5_std = calc_statistics_hdf5(training_data_path, batch_size=100, porcentage_dataset=dataset_percentage)

    # Initialize the normalizer with computed statistics
    normalizer = MyNormalize(h5_mean, h5_std, batch_size=100, porcentage_dataset=dataset_percentage)

    # Apply normalization to the training dataset and save the results
    normalizer.normalize_hdf5(training_data_path, output_directory)

    # Apply normalization to the validation dataset and save the results
    normalizer.normalize_hdf5(validation_data_path, output_directory)


# Execute the script
if __name__ == "__main__":
    main()

