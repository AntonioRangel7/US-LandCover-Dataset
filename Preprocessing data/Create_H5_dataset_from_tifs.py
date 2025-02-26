"""
Authors: Antonio Rangel & Juan Terven
Date: 2023

Description:
This script creates an HDF5 dataset from Sentinel-2 and Dynamic World composite images.
It processes raster images, extracts synchronized tiles, and stores them efficiently in an HDF5 file
for training deep learning models.

Main Features:
- Extracts and synchronizes tiles from Sentinel-2 and Dynamic World composite images.
- Applies filtering based on NaN thresholds to exclude invalid tiles.
- Crops images into fixed-size tiles (default: 256x256 pixels).
- Saves processed data in an HDF5 format with compression for efficient storage.
- Ensures consistency between mask and predictor datasets by matching file patterns.
- Implements batch processing to handle large datasets efficiently.
- Provides interactive CLI for dataset name, batch size, and execution confirmation.
"""

import rasterio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
import h5py

def extract_pattern(file_path):
    """
    Extracts a specific date pattern from the given file path.

    Args:
        file_path (str or Path): The file path containing the pattern.

    Returns:
        str: Extracted pattern if found, otherwise None.
    """
    if not isinstance(file_path, (str, Path)):
        return None  # Ignore if not a string or Path object
    file_path = str(file_path)  # Convert Path object to string
    pattern = r"(\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2} Part\d+)"
    match = re.search(pattern, file_path)
    return match.group(1) if match else None


def wait_for_key():
    """
    Waits for user input to either continue or exit the program.

    Returns:
        bool: True if the user decides to continue, False if they choose to exit.
    """
    while True:
        choice = input("Press 'Enter' to continue or type 'esc' to exit: ").strip().lower()
        if choice == "":
            print("Continuing execution.")
            return True
        elif choice == "esc":
            print("Execution aborted.")
            return False


def crop_and_sync_tiles(mask_file_paths, predictor_file_paths, tile_size=256):
    """
    Crops and synchronizes tiles from given mask and predictor images.

    Args:
        mask_file_paths (list): List of file paths for mask images.
        predictor_file_paths (list): List of file paths for predictor images.
        tile_size (int): Size of each tile to be extracted. Default is 256.

    Returns:
        tuple: Two lists containing cropped mask tiles and predictor tiles.
    """
    all_tiles_masks = []
    all_tiles_predictors = []
    nan_threshold = 0.2  # Maximum allowed NaN percentage

    for mask_file_path, predictor_file_path in zip(mask_file_paths, predictor_file_paths):
        with rasterio.open(mask_file_path) as mask_dataset, rasterio.open(predictor_file_path) as predictor_dataset:
            mask_image = mask_dataset.read(1)
            predictor_image = predictor_dataset.read()

            if mask_image.shape != predictor_image.shape[1:]:
                print(f"Dimension mismatch between mask and predictors: {mask_file_path}")
                continue

            height, width = mask_image.shape

            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    mask_tile = mask_image[i:i + tile_size, j:j + tile_size]
                    predictor_tile = predictor_image[:, i:i + tile_size, j:j + tile_size]

                    if (mask_tile.shape == (tile_size, tile_size) and
                            predictor_tile.shape == (3, tile_size, tile_size) and
                            (np.isnan(mask_tile).sum() / mask_tile.size) <= nan_threshold and
                            (np.isnan(predictor_tile).sum() / predictor_tile.size) <= nan_threshold):
                        mask_tile = np.nan_to_num(mask_tile, nan=0)
                        predictor_tile = np.nan_to_num(predictor_tile, nan=0)

                        all_tiles_masks.append(mask_tile)
                        all_tiles_predictors.append(predictor_tile)

    return all_tiles_masks, all_tiles_predictors


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


def main():
    dataset_name = input("Enter dataset name: ")
    base_directory = "<BASE_DIRECTORY_PATH>/"
    hdf5_output_path = f"{base_directory}{dataset_name}_masks_and_predictors.h5"

    # Retrieve file paths for Dynamic World and Sentinel-2 composite images
    dynamic_world_files = sorted(Path(base_directory + f"DW/{dataset_name}/").glob('Dynamic world COMPOSITE*.tif'))
    sentinel2_files = sorted(Path(base_directory + f"SENTINEL/S2_{dataset_name}/").glob('SENTINEL COMPOSITE*.tif'))

    # Validate if lists have different lengths
    if len(dynamic_world_files) != len(sentinel2_files):
        dw_patterns = [extract_pattern(file) for file in dynamic_world_files]
        s2_patterns = [extract_pattern(file) for file in sentinel2_files]
        common_patterns = set(dw_patterns) & set(s2_patterns)

        dynamic_world_files = [file for file in dynamic_world_files if extract_pattern(file) in common_patterns]
        sentinel2_files = [file for file in sentinel2_files if extract_pattern(file) in common_patterns]

        print(f"{len(sentinel2_files)} masks have been created.")

    # Confirm whether to proceed or abort
    if not wait_for_key():
        print("The program has been aborted.")
        return

    # Process batches
    batch_size = int(input("Enter batch size: "))
    for batch_start in tqdm(range(0, len(dynamic_world_files), batch_size), desc="Processing batches"):
        batch_end = batch_start + 100
        dw_batch = dynamic_world_files[batch_start:batch_end]
        s2_batch = sentinel2_files[batch_start:batch_end]

        # Crop and synchronize tiles
        masks, predictors = crop_and_sync_tiles(dw_batch, s2_batch, tile_size=256)

        # Save batch results in HDF5 format
        save_batch_to_hdf5(hdf5_output_path, np.array(masks), np.array(predictors), append=True)

    print("The program has successfully completed.")


# Execute the script
if __name__ == "__main__":
    main()

