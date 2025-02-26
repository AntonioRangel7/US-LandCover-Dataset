"""
Authors: Antonio Rangel & Juan Terven
Date: 2023

Description:
This script applies the Mosaic data augmentation technique to an HDF5 dataset for semantic segmentation.
It processes RGB images and their corresponding masks, generating augmented samples to balance class distributions.

Main Features:
- Combines RGB images and corresponding masks into four-channel tensors.
- Resizes images and adjusts bounding box annotations to maintain consistency.
- Generates bounding boxes from class masks for object detection tasks.
- Implements the Mosaic augmentation strategy by merging four images into a single composite.
- Identifies and extracts minority class tiles based on class distribution thresholds.
- Processes large datasets stored in HDF5 format, allowing efficient batch operations.
- Applies data augmentation with controlled parameters and stores augmented samples in HDF5.
- Provides a command-line interface for user-defined dataset processing parameters.
"""


import h5py
import torch
from math import sqrt
from random import uniform
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import argparse

def combine_rgb_images_and_masks(image_list, mask_list):
    """
    Combines RGB images and corresponding masks into a single tensor with 4 channels.

    Args:
        image_list (list of np.ndarray): List of images in shape (H, W, 3) for RGB.
        mask_list (list of np.ndarray): List of masks in shape (H, W).

    Returns:
        torch.Tensor: Combined tensor with shape (N, 4, H, W), where C = 4 (RGB + mask).
    """
    combined_tensors = []

    for image, mask in zip(image_list, mask_list):
        mask = np.expand_dims(mask, axis=-1)  # Convert (H, W) -> (H, W, 1)
        combined_image = np.concatenate((image, mask), axis=-1)  # Shape (H, W, 4)
        combined_tensor = torch.tensor(combined_image.transpose(2, 0, 1), dtype=torch.float32)
        combined_tensors.append(combined_tensor)

    return torch.stack(combined_tensors)


def resize_image_and_annotations(image, annotations, target_size):
    """
    Resizes an image and its annotations to a specified size.

    Args:
        image (np.ndarray): Image in numpy format with shape (H, W, RGB).
        annotations (dict): Dictionary containing bounding box annotations.
        target_size (tuple): Target width and height for resizing.

    Returns:
        tuple: Resized image and updated annotations.
    """
    image = Image.fromarray(image)
    scale_w = target_size[0] / image.width
    scale_h = target_size[1] / image.height
    image = image.resize((target_size[0], target_size[1]))
    image = np.array(image, dtype=np.uint8).transpose(2, 0, 1)

    for i in range(len(annotations["bbox"])):
        annotations["bbox"][i][0] *= scale_w
        annotations["bbox"][i][2] *= scale_w
        annotations["bbox"][i][1] *= scale_h
        annotations["bbox"][i][3] *= scale_h

    return image, annotations


def generate_bounding_boxes_from_mask(mask, num_classes):
    """
    Generates bounding boxes from a class mask.

    Args:
        mask (np.ndarray): Mask image where each pixel value represents a class.
        num_classes (int): Number of distinct classes in the mask.

    Returns:
        tuple: List of bounding boxes (x, y, w, h) and corresponding class labels.
    """
    bounding_boxes = []
    class_labels = []

    for class_id in range(num_classes):
        class_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append([x, y, w, h])
            class_labels.append(class_id)

    return bounding_boxes, class_labels


def generate_annotations_from_masks(mask_list, num_classes):
    """
    Generates annotations (bounding boxes, class labels, and pixel-class map) for multiple masks.

    Args:
        mask_list (list of np.ndarray): List of masks where each pixel represents a class.
        num_classes (int): Number of distinct classes in each mask.

    Returns:
        list: List of annotation dictionaries containing bounding boxes, class labels, and pixel-class maps.
    """
    annotations = []

    for mask in mask_list:
        bounding_boxes, class_labels = generate_bounding_boxes_from_mask(mask, num_classes)
        pixel_class_map = torch.tensor(mask, dtype=torch.int16)
        annotation = {"bbox": bounding_boxes, "cls": class_labels, "pix_cls": pixel_class_map}
        annotations.append(annotation)

    return annotations


def Mosaic(images, annotations, output_size):
    """
    Applies the Mosaic data augmentation technique by combining four images of different sizes.

    Steps:
    1. Resize the images to the specified output size.
    2. Arrange the four images into a single combined image, positioning each in a different quadrant.
    3. Adjust bounding boxes to their correct locations within the combined image.
    4. Extract a random cutout of the final size from the combined image.
    5. Remove annotations that fall outside the cutout.
    6. Adjust bounding boxes that are partially within the cutout.

    Args:
        images (list of torch.Tensor): List of four input images.
        annotations (list of dict): List of bounding box annotations for each image.
        output_size (tuple): Desired output size (height, width).

    Returns:
        tuple: Augmented image and corresponding mask.
    """
    # Resize images to match the output size
    for i in range(len(images)):
        if images[i].shape[1] != output_size[0] or images[i].shape[2] != output_size[1]:
            images[i], annotations[i] = resize_image_and_annotations(images[i], annotations[i], output_size)

    # Create a tensor to hold the combined image (double the output size)
    combined_image = torch.zeros((4, output_size[0] * 2, output_size[1] * 2), requires_grad=False)

    # Place each image in its corresponding quadrant
    combined_image[:, :output_size[0], :output_size[1]] = images[0].clone().detach()
    combined_image[:, :output_size[0], output_size[1]:] = images[1].clone().detach()
    combined_image[:, output_size[0]:, :output_size[1]] = images[2].clone().detach()
    combined_image[:, output_size[0]:, output_size[1]:] = images[3].clone().detach()

    # Adjust bounding boxes for new positions
    adjusted_bboxes = []
    adjusted_classes = []

    for idx, (annotation, x_offset, y_offset) in enumerate(
            zip(annotations, [0, output_size[0], 0, output_size[0]], [0, 0, output_size[1], output_size[1]])):
        adjusted_classes += annotation["cls"]
        for bbox in annotation["bbox"]:
            adjusted_bboxes.append([bbox[0] + x_offset, bbox[1] + y_offset, bbox[2], bbox[3]])

    adjusted_bboxes = np.array(adjusted_bboxes, dtype=np.uint16)
    adjusted_classes = np.array(adjusted_classes, dtype=np.uint16)

    # Generate a random crop
    crop_x = round(uniform(sqrt(output_size[0]), output_size[0] - sqrt(output_size[0])))
    crop_y = round(uniform(sqrt(output_size[1]), output_size[1] - sqrt(output_size[1])))
    crop_bounds_x = (crop_x, crop_x + output_size[0])
    crop_bounds_y = (crop_y, crop_y + output_size[1])
    cropped_image = combined_image[:, crop_bounds_y[0]:crop_bounds_y[1], crop_bounds_x[0]:crop_bounds_x[1]]

    # Remove annotations outside the cropped region
    xA = np.maximum(crop_bounds_x[0], adjusted_bboxes[:, 0])
    yA = np.maximum(crop_bounds_y[0], adjusted_bboxes[:, 1])
    xB = np.minimum(crop_bounds_x[1], adjusted_bboxes[:, 0] + adjusted_bboxes[:, 2])
    yB = np.minimum(crop_bounds_y[1], adjusted_bboxes[:, 1] + adjusted_bboxes[:, 3])

    intersection_area = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    valid_indices = intersection_area != 0

    adjusted_bboxes = adjusted_bboxes[valid_indices]
    adjusted_classes = adjusted_classes[valid_indices]

    # Adjust bounding boxes to fit within the cropped image
    adjusted_bboxes[:, 0] = np.clip(adjusted_bboxes[:, 0] - crop_bounds_x[0], 0, output_size[0])
    adjusted_bboxes[:, 1] = np.clip(adjusted_bboxes[:, 1] - crop_bounds_y[0], 0, output_size[1])
    adjusted_bboxes[:, 2] = np.clip(adjusted_bboxes[:, 2], 0, output_size[0] - adjusted_bboxes[:, 0])
    adjusted_bboxes[:, 3] = np.clip(adjusted_bboxes[:, 3], 0, output_size[1] - adjusted_bboxes[:, 1])

    # Create pixel-class map
    pixel_class_map = torch.zeros(cropped_image.shape[1:], dtype=torch.int16, requires_grad=False)
    for i in range(len(adjusted_bboxes)):
        bbox = adjusted_bboxes[i]
        pixel_class_map[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = adjusted_classes[i]

    new_annotations = {"bbox": adjusted_bboxes.tolist(), "cls": adjusted_classes, "pix_cls": pixel_class_map}
    predictors = cropped_image[:3, :, :].permute(1, 2, 0)
    mask = cropped_image[3:, :, :].squeeze()

    return predictors, mask


def identify_minority_tiles_multi_class(train_predictors, train_masks, classes, threshold=0.05):
    """
    Identifies tiles where the proportion of specified classes exceeds a given threshold.

    Args:
        train_predictors (np.ndarray): Array of predictor images.
        train_masks (np.ndarray): Corresponding array of mask images.
        classes (list): List of class labels to consider as minority.
        threshold (float): Minimum proportion of the class required to select a tile.

    Returns:
        tuple: Arrays of selected predictor and mask tiles.
    """
    minority_tiles = []
    minority_masks = []

    for predictor, mask in zip(train_predictors, train_masks):
        for class_index in classes:
            class_pixels = np.sum(mask == class_index)
            total_pixels = mask.size
            class_proportion = class_pixels / total_pixels

            if class_proportion >= threshold:
                minority_tiles.append(predictor)
                minority_masks.append(mask)
                break  # Avoid duplicate selection if multiple classes match

    return np.array(minority_tiles), np.array(minority_masks)


def identify_minority_tiles_hdf5(file_name, batch_size, threshold=0.05, classes=[0, 1], dataset_percentage=1):
    """
    Identifies minority tiles from an HDF5 dataset based on class proportions.

    Args:
        file_name (str): Path to the HDF5 dataset.
        batch_size (int): Number of samples to process per batch.
        threshold (float): Minimum proportion of the class required to select a tile.
        classes (list): List of class labels to consider as minority.
        dataset_percentage (float): Percentage of the dataset to process.

    Returns:
        tuple: Arrays of selected predictor and mask tiles.
    """
    minority_tiles = []
    minority_masks = []

    with h5py.File(file_name, 'r') as f:
        total_elements = int(f['masks'].shape[0] * dataset_percentage)

        for start in tqdm(range(0, total_elements, batch_size), desc="Processing batches"):
            end = min(start + batch_size, total_elements)
            predictors_batch = f['predictors'][start:end]
            masks_batch = f['masks'][start:end]

            selected_tiles, selected_masks = identify_minority_tiles_multi_class(predictors_batch, masks_batch, classes,
                                                                                 threshold)

            if selected_tiles.size > 0 and selected_masks.size > 0:
                minority_tiles.append(selected_tiles)
                minority_masks.append(selected_masks)

    if minority_tiles:
        all_minority_tiles = np.concatenate(minority_tiles, axis=0)
        all_minority_masks = np.concatenate(minority_masks, axis=0)
    else:
        all_minority_tiles = np.empty((0,))
        all_minority_masks = np.empty((0,))

    print(f"Total selected tiles: {all_minority_tiles.shape[0]}")
    return all_minority_tiles, all_minority_masks


def save_batch_to_hdf5(file_name, masks, predictors, append=False):
    """
    Saves batches of masks and predictors to an HDF5 file.

    Args:
        file_name (str): Path to the HDF5 file.
        masks (np.ndarray): Array of masks with shape (num_tiles, 256, 256).
        predictors (np.ndarray): Array of predictors with shape (num_tiles, 3, 256, 256).
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

def apply_data_augmentation(file_name, predictor_tiles, mask_tiles, num_augmentations=3):
    """
    Applies data augmentation to the given predictor and mask tiles.

    Args:
        file_name (str): Path to the HDF5 file where augmented data will be saved.
        predictor_tiles (list): List of predictor image tiles.
        mask_tiles (list): List of corresponding mask image tiles.
        num_augmentations (int): Number of augmentations to perform.
    """
    augmented_predictors = []
    augmented_masks = []
    saved_tiles = 0

    for _ in tqdm(range(num_augmentations), desc="Applying Data Augmentation"):
        for idx in range(len(predictor_tiles) - 3):  # Ensure enough images
            predictors_batch = predictor_tiles[idx:idx + 4]
            masks_batch = mask_tiles[idx:idx + 4]
            predictors_batch = np.transpose(predictors_batch, (0, 2, 3, 1))
            annotations = generate_annotations_from_masks(masks_batch, 9)
            augmented_set = combine_rgb_images_and_masks(predictors_batch, masks_batch)
            augmented_predictor, augmented_mask = Mosaic(augmented_set[:4], annotations, size=[256, 256])
            augmented_predictors.append(augmented_predictor.numpy())
            augmented_masks.append(augmented_mask.squeeze(0).numpy())
            saved_tiles += 1

            if saved_tiles % 100 == 0:
                augmented_predictors_array = np.array(augmented_predictors)
                augmented_masks_array = np.array(augmented_masks)
                augmented_predictors_array = np.nan_to_num(augmented_predictors_array, nan=0)
                augmented_masks_array = np.nan_to_num(augmented_masks_array, nan=0)
                augmented_predictors_array = np.transpose(augmented_predictors_array, (0, 3, 1, 2))
                save_batch_to_hdf5(file_name, augmented_masks_array, augmented_predictors_array, append=True)
                augmented_predictors = []
                augmented_masks = []

        # Shuffle tiles
        indices = np.random.permutation(len(predictor_tiles))
        predictor_tiles = predictor_tiles[indices]
        mask_tiles = mask_tiles[indices]

    print(f"{saved_tiles} augmented tiles have been generated.")


def main(hdf5_dataset_path):
    """
    Main function to identify minority tiles and apply data augmentation.

    Args:
        hdf5_dataset_path (str): Path to the HDF5 dataset.
    """
    # Unified user input prompts
    batch_size = int(input("Enter batch size: "))
    dataset_percentage = float(input("Enter the percentage of the dataset to use (e.g., 1.0 for 100%): "))
    selected_classes = [int(value) for value in
                        input("Enter a list of integer class values separated by spaces: ").split()]
    num_augmentations = int(input("Enter the number of augmentations to perform: "))

    print(f"Batch size: {batch_size}")
    print(f"Dataset percentage: {dataset_percentage}")
    print(f"Selected classes: {selected_classes}")
    print(f"Number of augmentations: {num_augmentations}")

    # Identify minority tiles based on class distribution
    minority_tiles, minority_masks = identify_minority_tiles_hdf5(
        hdf5_dataset_path, batch_size, threshold=0.80, classes=selected_classes, dataset_percentage=dataset_percentage
    )

    # Apply data augmentation with mosaic augmentation enabled
    apply_data_augmentation(
        hdf5_dataset_path, minority_tiles, minority_masks, num_augmentations=num_augmentations
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process an HDF5 dataset.")
    parser.add_argument("dataset_path", help="Path to the dataset (e.g., TRAIN or TEST).")
    args = parser.parse_args()
    main(args.dataset_path)
