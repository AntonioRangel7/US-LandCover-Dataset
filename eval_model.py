"""
Authors: Antonio Rangel & Juan Terven
Date: 2023

Description:
This script evaluates a semantic segmentation model using either LRASPP or FCN architectures.
It processes test data stored in an HDF5 format and computes multiple evaluation metrics, including
F1-score, IoU, Precision, Recall, and Accuracy. The script is optimized for GPU inference.

Main Features:
- Supports evaluation of pre-trained LRASPP and FCN models.
- Loads test data from an HDF5 dataset for efficient processing.
- Computes per-class and overall evaluation metrics (IoU, F1-score, Precision, Recall, Accuracy).
- Provides GPU selection and monitoring using `nvidia-smi`.
- Implements batch-wise inference for scalable model evaluation.
- Measures total evaluation time for performance assessment.
- Interactive CLI for model selection and test dataset configuration.
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision.models.segmentation import lraspp_mobilenet_v3_large,LRASPP_MobileNet_V3_Large_Weights,fcn_resnet50,FCN_ResNet50_Weights
from sklearn.metrics import precision_score, recall_score, f1_score,jaccard_score
import subprocess
import torch
import h5py
from typing import List, Optional
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer


def nvidia_smi_function():
    """
    Executes the `nvidia-smi` command to check GPU status and availability.

    Prints the output of the command if successful, or an error message if not found.
    """
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output of nvidia-smi:")
        print(result.stdout)
    except FileNotFoundError:
        print("The command `nvidia-smi` was not found. Ensure NVIDIA drivers are installed.")


def ask_user(prompt="Enter a value: "):
    """
    Asks the user for input and returns the entered value.

    Args:
        prompt (str): The message displayed to the user.

    Returns:
        str: The user input.
    """
    return input(prompt)


def initialize_model(model_type, device, num_classes=9, seed=42):
    """
    Initializes and configures a segmentation model based on the specified type.

    Args:
        model_type (str): The type of model to initialize ('LRASPP' or 'FCN').
        device (torch.device): The device where the model will be loaded (CPU or GPU).
        num_classes (int): Number of output classes (default: 9).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        torch.nn.Module: The configured and loaded model.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if model_type == "LRASPP":
        model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT, num_classes=21)
        model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=1, stride=1)
        model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=1, stride=1)
        model_name = "BEST_LRASPP_MODEL.pth"
    elif model_type == "FCN":
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_classes=21)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        model_name = "BEST_FCN_MODEL.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'LRASPP' or 'FCN'.")

    model.load_state_dict(torch.load(model_name))
    model.to(device)
    return model


def setup_environment_and_model():
    """
    Sets up the environment and initializes the selected model.

    Prompts the user to select a model type and specify the dataset path.

    Returns:
        tuple: (model, device, test_data_path)
    """
    import os

    nvidia_smi_function()
    device = torch.device("cuda:0")
    print("Using GPU 1...")

    print("Select the model to evaluate:")
    print("0: LRASPP")
    print("1: FCN")
    model_choice = ask_user("Enter 0 or 1 for the desired model: ")
    model_type = "LRASPP" if model_choice == '0' else "FCN" if model_choice == '1' else "LRASPP"

    test_data_path = ask_user("Enter the path to your test data: ")
    model = initialize_model(model_type, device)

    return model, device, test_data_path


def accuracy_score_per_class(y, pred, num_classes):
    """
    Computes the accuracy for each class.

    Args:
        y (torch.Tensor): Ground truth labels.
        pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Accuracy per class.
    """
    class_counts = np.zeros(num_classes, dtype=float)
    class_accuracies = np.zeros(num_classes, dtype=float)

    for category in range(num_classes):
        category_mask = (y == category)
        if torch.any(category_mask):
            class_accuracies[category] = torch.count_nonzero(pred[category_mask] == category) / torch.count_nonzero(
                category_mask)

    return class_accuracies


def precision_score_per_class(y, pred, num_classes):
    """
    Computes the precision for each class.

    Args:
        y (torch.Tensor): Ground truth labels.
        pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Precision per class.
    """
    tp = np.zeros(num_classes, dtype=float)
    fp = np.zeros(num_classes, dtype=float)

    for category in range(num_classes):
        tp[category] += torch.count_nonzero((pred == category) & (y == category)).item()
        fp[category] += torch.count_nonzero((pred == category) & (y != category)).item()

    return np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)


def recall_score_per_class(y, pred, num_classes):
    """
    Computes the recall for each class.

    Args:
        y (torch.Tensor): Ground truth labels.
        pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Recall per class.
    """
    tp = np.zeros(num_classes, dtype=float)
    fn = np.zeros(num_classes, dtype=float)

    for category in range(num_classes):
        tp[category] += torch.sum((pred == category) & (y == category)).item()
        fn[category] += torch.sum((pred != category) & (y == category)).item()

    return np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)


def test_step_h5(h5_file, model: torch.nn.Module, batch_size: int):
    """
    Performs a test step using data stored in an HDF5 file.

    Args:
        h5_file (h5py.File): Open HDF5 file containing test data.
        model (torch.nn.Module): Trained PyTorch model for evaluation.
        batch_size (int): Number of samples per batch.

    Returns:
        None
    """
    model.cuda()
    batch_count = 0
    test_acc, test_f1, test_iou, test_precision, test_recall = 0.0, 0.0, 0.0, 0.0, 0.0
    test_recall_per_class, test_precision_per_class, test_accuracy_per_class = 0.0, 0.0, 0.0

    with torch.no_grad():
        total_samples = h5_file['masks'].shape[0]

        for start in tqdm(range(0, total_samples, batch_size), desc="Running test..."):
            end = min(start + batch_size, total_samples)

            predictors_batch = torch.tensor(h5_file['predictors'][start:end], dtype=torch.float32).cuda(
                non_blocking=True)
            masks_batch = torch.tensor(h5_file['masks'][start:end], dtype=torch.long).cuda(non_blocking=True)

            predictions = model(predictors_batch)['out'].argmax(dim=1)
            y_true = masks_batch.squeeze()

            batch_f1 = f1_score(y_true.cpu().numpy().flatten(), predictions.cpu().numpy().flatten(), average='macro')
            batch_precision = precision_score(y_true.cpu().numpy().flatten(), predictions.cpu().numpy().flatten(),
                                              average='macro', zero_division=0)
            batch_recall = recall_score(y_true.cpu().numpy().flatten(), predictions.cpu().numpy().flatten(),
                                        average='macro', zero_division=0)
            batch_accuracy = accuracy_score(y_true.cpu().numpy().flatten(), predictions.cpu().numpy().flatten())
            batch_iou = jaccard_score(y_true.cpu().numpy().flatten(), predictions.cpu().numpy().flatten(),
                                      average='macro')

            batch_recall_per_class = recall_score_per_class(y_true, predictions, 9)
            batch_precision_per_class = precision_score_per_class(y_true, predictions, 9)
            batch_accuracy_per_class = accuracy_score_per_class(y_true, predictions, 9)

            test_acc += batch_accuracy
            test_f1 += batch_f1
            test_precision += batch_precision
            test_recall += batch_recall
            test_iou += batch_iou
            test_recall_per_class += batch_recall_per_class
            test_precision_per_class += batch_precision_per_class
            test_accuracy_per_class += batch_accuracy_per_class
            batch_count += 1

        test_f1 /= batch_count
        test_acc /= batch_count
        test_iou /= batch_count
        test_precision /= batch_count
        test_recall /= batch_count
        test_recall_per_class /= batch_count
        test_precision_per_class /= batch_count
        test_accuracy_per_class /= batch_count

        print(
            f"Test F1-score: {test_f1:.2f} | Test Accuracy: {test_acc:.2f}\n Test Precision: {test_precision:.2f} | Test Recall: {test_recall:.2f}\n Test IoU: {test_iou:.2f}")

        num_classes = len(test_recall_per_class)
        print("Test recall per class:",
              ", ".join([f"Class {i}: {test_recall_per_class[i]:.3f}" for i in range(num_classes)]))
        print("Test precision per class:",
              ", ".join([f"Class {i}: {test_precision_per_class[i]:.3f}" for i in range(num_classes)]))
        print("Test accuracy per class:",
              ", ".join([f"Class {i}: {test_accuracy_per_class[i]:.3f}" for i in range(num_classes)]))


def evaluate_model(model: torch.nn.Module, batch_size: int, test_file: str):
    """
    Evaluates a trained model using an HDF5 test dataset.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        batch_size (int): Batch size for testing.
        test_file (str): Path to the HDF5 file containing test data.
        acc_fns (Optional[List]): List of additional accuracy functions (not currently used).

    Returns:
        None
    """
    with h5py.File(test_file, 'r', libver='latest', swmr=True) as h5_file:
        test_step_h5(h5_file, model, batch_size)


def main():
    """
    Main function to configure the environment, load the model, and evaluate it on test data.
    """
    # Configure environment and model
    model, device, test_data_path = setup_environment_and_model()

    # Start the timer to measure evaluation time
    start_time = timer()

    # Evaluate the model
    evaluate_model(
        model=model,
        batch_size=64,
        test_file=test_data_path
    )

    # End the timer and print execution time
    end_time = timer()
    print(f"Total evaluation time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()